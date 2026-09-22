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

"""Hardware-light gamepad teleoperator for the G1Ah robot.

Emits every key in `TELEOP_ACTION_KEYS` on every `get_action()` call: held body/hand
poses, D-pad-driven head targets, an RB/LB-blended hand open/close, and the 4
`REMOTE_AXES` driven by the sticks. All motion is time-based (rad/s, blend/s) so
behaviour does not depend on the calling loop's fps.
"""

from __future__ import annotations

import time
from functools import cached_property
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.robots.g1ah.g1ah_joints import (
    ALL_ACTION_KEYS,
    HEAD_LIMITS_RAD,
    TELEOP_ACTION_KEYS,
    default_action,
    hand_motor_names,
    hand_pose_rad,
)
from lerobot.robots.unitree_g1.g1_utils import REMOTE_AXES
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_g1ah_gamepad import G1AhGamepadTeleopConfig
from .gamepad_input import G1AhGamepadInput

_DT_CAP_S = 0.1


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class G1AhGamepadTeleop(Teleoperator):
    """Gamepad teleoperator emitting the full G1Ah teleop action space (45 keys)."""

    config_class = G1AhGamepadTeleopConfig
    name = "g1ah_gamepad"

    def __init__(self, config: G1AhGamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.gamepad: G1AhGamepadInput | None = None

        self._target: dict[str, float] = default_action()
        if config.initial_positions is not None:
            invalid = set(config.initial_positions) - set(ALL_ACTION_KEYS)
            if invalid:
                raise ValueError(f"Unknown initial_positions keys: {sorted(invalid)}")
            self._target.update(config.initial_positions)

        self._hand_blend: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._last_t: float | None = None
        self._open = {side: hand_pose_rad(side, False) for side in ("left", "right")}
        self._closed = {side: hand_pose_rad(side, True) for side in ("left", "right")}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(TELEOP_ACTION_KEYS, float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        self.gamepad = G1AhGamepadInput(self.config.layout, self.config.deadzone)
        self.gamepad.start()
        self._last_t = None
        print("G1Ah gamepad controls:")
        print("  D-pad: head pan (left/right) / tilt (up/down)")
        print("  RB / LB: hold to close right / left hand")
        print("  Sticks: locomotion command (remote.lx/ly/rx/ry)")
        print("  Y / A / X: end episode success / failure / rerecord")

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
        pass

    def _step_head(self, dt: float) -> None:
        hx, hy = self.gamepad.hat()
        pan = self._target["xl330_joint.q"] - hx * self.config.head_speed_rad_s * dt
        tilt_sign = -1.0 if self.config.invert_tilt else 1.0
        tilt = self._target["d455_joint.q"] + tilt_sign * hy * self.config.head_speed_rad_s * dt
        pan_lo, pan_hi = HEAD_LIMITS_RAD["xl330_joint"]
        tilt_lo, tilt_hi = HEAD_LIMITS_RAD["d455_joint"]
        self._target["xl330_joint.q"] = min(max(pan, pan_lo), pan_hi)
        self._target["d455_joint.q"] = min(max(tilt, tilt_lo), tilt_hi)

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
            for name, open_rad, closed_rad in zip(
                hand_motor_names(side), self._open[side], self._closed[side], strict=True
            ):
                self._target[f"{name}.q"] = _lerp(open_rad, closed_rad, blend)

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

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.perf_counter()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, _DT_CAP_S)
        self._last_t = now

        self.gamepad.update()
        self._step_head(dt)
        self._step_hands(dt)
        remote = self._remote_axes()

        return {**self._target, **remote}

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
