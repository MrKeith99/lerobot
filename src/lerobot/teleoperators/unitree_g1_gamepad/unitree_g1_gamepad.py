#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware-light gamepad teleoperator for the Unitree G1, whatever its end effector and head.

Emits every key in `TELEOP_ACTION_KEYS` on every `get_action()` call: held body poses (taken from
the robot's first observation via `send_feedback`, else zeros), D-pad-driven pan/tilt head targets,
an RB/LB-blended hand open/close (`k{Side}Hand.closure` in [0, 1] plus the matching joint targets of
every end effector), the 4 `REMOTE_AXES` driven by the sticks (also emitted as the
`kNavVx/Vy/YawRate.cmd` navigation command) and the 16 `REMOTE_BUTTONS`, of which the L2/R2
triggers drive the locomotion controller's waist raise/lower slots. The robot and the dataset only
keep the keys its embodiment has. All motion is time-based (rad/s, blend/s) so behaviour does not
depend on the calling loop's fps.
"""

from __future__ import annotations

import time
from functools import cached_property
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.robots.unitree_g1.end_effectors import (
    ALL_HAND_KEYS,
    HAND_CLOSURE_KEYS,
    HAND_SIDES,
    HAND_SPECS,
    hand_closure_key,
)
from lerobot.robots.unitree_g1.g1_utils import (
    BASE_HEIGHT_KEY,
    BODY_KEYS,
    GROOT_BASE_HEIGHT_DEFAULT,
    GROOT_BASE_HEIGHT_RANGE,
    GROOT_BASE_HEIGHT_RATE,
    NAV_KEYS,
    REMOTE_AXES,
    REMOTE_BUTTONS,
    REMOTE_KEYS,
    nav_from_remote,
)
from lerobot.robots.unitree_g1.heads import DEFAULT_HEAD_Q, HEAD_KEYS, HEAD_LIMITS_RAD
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_unitree_g1_gamepad import UnitreeG1GamepadTeleopConfig
from .gamepad_input import UnitreeG1GamepadInput

_DT_CAP_S = 0.1
# Joint targets: the 29-slot body, the pan/tilt head and every end effector's joints.
TARGET_KEYS: tuple[str, ...] = BODY_KEYS + HEAD_KEYS + ALL_HAND_KEYS
TELEOP_ACTION_KEYS: tuple[str, ...] = (
    TARGET_KEYS + HAND_CLOSURE_KEYS + REMOTE_KEYS + NAV_KEYS + (BASE_HEIGHT_KEY,)
)
# GrootLocomotionController reads waist raise/lower from remote.button.0/4 (the wireless
# remote's R1/R2 slots); R1 is the hand button here, so L2 stands in for the raise slot.
_WAIST_RAISE_KEY = "remote.button.0"
_WAIST_LOWER_KEY = "remote.button.4"


def default_targets() -> dict[str, float]:
    """Zero body, centered head, open hands."""
    targets = dict.fromkeys(BODY_KEYS, 0.0)
    targets.update(zip(HEAD_KEYS, DEFAULT_HEAD_Q, strict=True))
    for spec in HAND_SPECS.values():
        for side in HAND_SIDES:
            targets.update(zip(spec.joint_keys(side), spec.open_q[side], strict=True))
    return targets


class UnitreeG1GamepadTeleop(Teleoperator):
    """Gamepad teleoperator emitting the full Unitree G1 teleop action space (`TELEOP_ACTION_KEYS`)."""

    config_class = UnitreeG1GamepadTeleopConfig
    name = "unitree_g1_gamepad"

    def __init__(self, config: UnitreeG1GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.gamepad: UnitreeG1GamepadInput | None = None

        self._target: dict[str, float] = {
            **default_targets(),
            **dict.fromkeys(HAND_CLOSURE_KEYS, 0.0),
            BASE_HEIGHT_KEY: GROOT_BASE_HEIGHT_DEFAULT,
        }
        if config.initial_positions is not None:
            invalid = (
                set(config.initial_positions) - set(TARGET_KEYS) - set(HAND_CLOSURE_KEYS) - {BASE_HEIGHT_KEY}
            )
            if invalid:
                raise ValueError(f"Unknown initial_positions keys: {sorted(invalid)}")
            self._target.update(config.initial_positions)

        self._hand_blend: dict[str, float] = {
            side: min(max(self._target[hand_closure_key(side)], 0.0), 1.0) for side in HAND_SIDES
        }
        self._last_t: float | None = None
        self._synced_to_robot = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(TELEOP_ACTION_KEYS, float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        self.gamepad = UnitreeG1GamepadInput(self.config.layout, self.config.deadzone)
        self.gamepad.start()
        self._last_t = None
        if self.config.preset == "xbox":
            rb_lb, lt_rt, yax = "RB / LB", "LT / RT", "Y / A / X"
        else:
            rb_lb, lt_rt, yax = "R1 / L1", "L2 / R2", "Triangle / Cross / Square"
        print(f"Unitree G1 gamepad controls ({self.config.preset}):")
        print("  D-pad: head pan (left/right) / tilt (up/down)")
        print(f"  {rb_lb}: hold to close right / left hand")
        print(f"  {lt_rt}: hold to raise / lower waist (GrootLocomotionController)")
        print("  Sticks: navigation command (kNavVx/Vy/YawRate.cmd, from remote.lx/ly/rx)")
        print(f"  {yax}: end episode success / failure / rerecord")

    @property
    def is_connected(self) -> bool:
        return self.gamepad is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Start from the robot's measured pose: the first observation with every body joint sets the
        body/head targets and the hand closures, except keys pinned by `initial_positions`."""
        if self._synced_to_robot or not all(key in feedback for key in BODY_KEYS):
            return
        pinned = set(self.config.initial_positions or ())
        for key in BODY_KEYS + HEAD_KEYS:
            if key in feedback and key not in pinned:
                self._target[key] = float(feedback[key])
        for side in HAND_SIDES:
            closure = self._feedback_closure(feedback, side)
            if closure is not None and hand_closure_key(side) not in pinned:
                self._hand_blend[side] = min(max(closure, 0.0), 1.0)
        self._synced_to_robot = True

    @staticmethod
    def _feedback_closure(feedback: dict[str, Any], side: str) -> float | None:
        """A hand's closure from the robot's closure key, else from the joints of its end effector."""
        key = hand_closure_key(side)
        if key in feedback:
            return float(feedback[key])
        for spec in HAND_SPECS.values():
            motor_keys = spec.joint_keys(side)
            if all(motor_key in feedback for motor_key in motor_keys):
                return spec.q_to_closure(side, [float(feedback[motor_key]) for motor_key in motor_keys])
        return None

    def _step_head(self, dt: float) -> None:
        hx, hy = self.gamepad.hat()
        yaw_key, pitch_key = HEAD_KEYS
        pan = self._target[yaw_key] - hx * self.config.head_speed_rad_s * dt
        tilt_sign = -1.0 if self.config.invert_tilt else 1.0
        tilt = self._target[pitch_key] + tilt_sign * hy * self.config.head_speed_rad_s * dt
        pan_lo, pan_hi = HEAD_LIMITS_RAD["kHeadYaw"]
        tilt_lo, tilt_hi = HEAD_LIMITS_RAD["kHeadPitch"]
        self._target[yaw_key] = min(max(pan, pan_lo), pan_hi)
        self._target[pitch_key] = min(max(tilt, tilt_lo), tilt_hi)

    def _step_hands(self, dt: float) -> None:
        pressed = {
            "right": self.gamepad.button(self.config.layout.button_rb),
            "left": self.gamepad.button(self.config.layout.button_lb),
        }
        max_step = self.config.hand_blend_per_s * dt
        for side, is_pressed in pressed.items():
            target_blend = 1.0 if is_pressed else 0.0
            current = self._hand_blend[side]
            delta = target_blend - current
            step = max(-max_step, min(max_step, delta))
            self._hand_blend[side] = current + step
            blend = self._hand_blend[side]
            self._target[hand_closure_key(side)] = blend
            for spec in HAND_SPECS.values():
                self._target.update(zip(spec.joint_keys(side), spec.closure_to_q(side, blend), strict=True))

    def _remote_axes(self) -> dict[str, float]:
        if not self.config.emit_remote_axes:
            return dict.fromkeys(REMOTE_AXES, 0.0)
        layout = self.config.layout
        return {
            "remote.lx": self.gamepad.axis(layout.left_x),
            "remote.ly": -self.gamepad.axis(layout.left_y),
            "remote.rx": self.gamepad.axis(layout.right_x),
            "remote.ry": -self.gamepad.axis(layout.right_y),
        }

    def _step_base_height(self, dt: float) -> None:
        layout = self.config.layout
        raise_ = self.gamepad.axis(layout.trigger_left) > 0.0
        lower = self.gamepad.axis(layout.trigger_right) > 0.0
        lo, hi = GROOT_BASE_HEIGHT_RANGE
        height = self._target[BASE_HEIGHT_KEY] + GROOT_BASE_HEIGHT_RATE * dt * (raise_ - lower)
        self._target[BASE_HEIGHT_KEY] = min(max(height, lo), hi)

    def _remote_buttons(self) -> dict[str, float]:
        layout = self.config.layout
        buttons = dict.fromkeys(REMOTE_BUTTONS, 0.0)
        buttons[_WAIST_RAISE_KEY] = float(self.gamepad.axis(layout.trigger_left) > 0.0)
        buttons[_WAIST_LOWER_KEY] = float(self.gamepad.axis(layout.trigger_right) > 0.0)
        return buttons

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.perf_counter()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, _DT_CAP_S)
        self._last_t = now

        self.gamepad.update()
        self._step_head(dt)
        self._step_hands(dt)
        self._step_base_height(dt)

        remote_axes = self._remote_axes()
        nav = nav_from_remote(remote_axes["remote.lx"], remote_axes["remote.ly"], remote_axes["remote.rx"])
        return {**self._target, **remote_axes, **nav, **self._remote_buttons()}

    def get_teleop_events(self) -> dict[str, Any]:
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        self.gamepad.update()
        is_intervention = self.gamepad.should_intervene()
        episode_end_status = self.gamepad.get_episode_end_status()
        terminate_episode = episode_end_status in (
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        )
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None
