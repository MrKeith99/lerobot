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

"""XR headset teleoperator for the Unitree G1: a thin adapter over the xr_teleoperate fork's core.

The fork reads the headset (televuer), solves the arm IK on the sim's composed model of this embodiment
and retargets the hands; this adapter turns that into the gamepad teleop's keys: arm `.q` (other body
joints held at the robot's first observed pose), `kHeadYaw/Pitch.q` with a pan/tilt head, the hand
closure plus its joint targets, the navigation command and remote axes, and `kBaseHeight.cmd`.

Controllers: A engages/disengages (recentering on the operator's facing direction and starting from the
measured arm pose), B stops (hold every target, no walking) until A re-engages, triggers close the
hands, the left stick walks, the right stick's x turns, X/Y held lower/raise the base, and the
right/left stick clicks end the episode as success/rerecord. Hand tracking has no buttons: it engages
once, when tracking first starts. Targets hold while tracking is lost.
"""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.robots.unitree_g1.end_effectors import (
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
    ROBOT_TYPE_FEEDBACK_KEY,
    nav_from_remote,
)
from lerobot.robots.unitree_g1.heads import DEFAULT_HEAD_Q, HEAD_KEYS, HEAD_LIMITS_RAD
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_unitree_g1_xr import UnitreeG1XRTeleopConfig

logger = logging.getLogger(__name__)

_DT_CAP_S = 0.1
# GrootLocomotionController's waist raise/lower slots, as in the gamepad teleop.
_WAIST_RAISE_KEY = "remote.button.0"
_WAIST_LOWER_KEY = "remote.button.4"


def arm_key(joint_name: str) -> str:
    """`left_shoulder_pitch_joint` -> `kLeftShoulderPitch.q`."""
    words = joint_name.removesuffix("_joint").split("_")
    return "k" + "".join(word.capitalize() for word in words) + ".q"


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _step_toward(current: float, target: float, max_step: float) -> float:
    return current + _clip(target - current, -max_step, max_step)


class UnitreeG1XRTeleop(Teleoperator):
    """XR teleoperator emitting the Unitree G1 teleop keys of one embodiment."""

    config_class = UnitreeG1XRTeleopConfig
    name = "unitree_g1_xr"

    def __init__(self, config: UnitreeG1XRTeleopConfig, xr_input: Any = None, arm_ik: Any = None):
        super().__init__(config)
        self.config = config
        self.hand_spec = HAND_SPECS.get(config.end_effector)
        self.xr = xr_input
        self.arm_ik = arm_ik
        self._retarget = None
        self._head = config.head_mount == "pan_tilt"

        self._target: dict[str, float] = dict.fromkeys(BODY_KEYS, 0.0)
        self._target[BASE_HEIGHT_KEY] = GROOT_BASE_HEIGHT_DEFAULT
        if self._head:
            self._target.update(zip(HEAD_KEYS, DEFAULT_HEAD_Q, strict=True))
        for side in HAND_SIDES if self.hand_spec is not None else ():
            self._set_closure(side, 0.0)

        self._engaged = False
        self._auto_engaged = False
        self._robot_type_checked = False
        self._synced_to_robot = False
        self._measured: dict[str, float] = {}
        self._buttons: dict[str, bool] = {}
        self._pending_events: set[str] = set()
        self._last_t: float | None = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        keys = BODY_KEYS + (HEAD_KEYS if self._head else ())
        if self.hand_spec is not None:
            keys += HAND_CLOSURE_KEYS + self.hand_spec.joint_keys()
        return dict.fromkeys(keys + REMOTE_KEYS + NAV_KEYS + (BASE_HEIGHT_KEY,), float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.xr is not None and self.arm_ik is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @property
    def engaged(self) -> bool:
        return self._engaged

    def connect(self, calibrate: bool = True) -> None:
        require_package("xr_teleoperate", extra="unitree_g1_xr")
        from xr_teleoperate.core import G1ArmIK, XRInput, make_hand_retargeter

        cfg = self.config
        self._retarget = make_hand_retargeter(cfg.end_effector)
        if self.arm_ik is None:
            self.arm_ik = G1ArmIK(cfg.body, cfg.end_effector, sim_root=cfg.sim_root)
        if self.xr is None:
            self.xr = XRInput(
                input_mode=cfg.input_mode,
                display_mode=cfg.display_mode,
                cert_file=cfg.cert_file,
                key_file=cfg.key_file,
            )
        self._last_t = None
        print(f"Unitree G1 XR teleop ({cfg.body}, {cfg.end_effector}, {cfg.head_mount}, {cfg.head_sensor}):")
        print("  Headset browser: https://<this host>:8012 (or https://vuer.ai?ws=wss://<this host>:8012)")
        print("  A: engage / disengage (face the robot's forward direction first)   B: stop")
        print("  Triggers: close hands   Left stick: walk   Right stick x: turn   X / Y: lower / raise base")
        print("  Right / left stick click: end episode success / rerecord")

    def disconnect(self) -> None:
        if self.xr is not None and hasattr(self.xr, "close"):
            self.xr.close()
        self.xr = None
        self._engaged = False

    def _set_closure(self, side: str, closure: float) -> None:
        self._target[hand_closure_key(side)] = closure
        self._target.update(
            zip(self.hand_spec.joint_keys(side), self.hand_spec.closure_to_q(side, closure), strict=True)
        )

    def _check_embodiment(self, feedback: dict[str, Any]) -> None:
        cfg = self.config
        robot_type = feedback.get(ROBOT_TYPE_FEEDBACK_KEY)
        if robot_type is not None and not self._robot_type_checked:
            parts = str(robot_type).split("-")
            got = (parts[1].split("_", 1)[0], *parts[2:]) if len(parts) == 5 else tuple(parts)
            want = (cfg.body, cfg.end_effector, cfg.head_mount, cfg.head_sensor)
            if parts[0] != "unitree_g1" or got != want:
                raise ValueError(
                    f"The XR teleop is set up for {want} (body, end_effector, head_mount, head_sensor) but the "
                    f"robot is {robot_type!r}; pass the robot's values as --teleop.*"
                )
            self._robot_type_checked = True
        if self._head != all(key in feedback for key in HEAD_KEYS):
            raise ValueError(
                f"The XR teleop has head_mount={cfg.head_mount!r} but the robot "
                f"{'has no' if self._head else 'has a'} pan/tilt head"
            )

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Check the embodiment, track the measured arm pose and, on the first full observation, hold every
        target at the robot's pose."""
        if not all(key in feedback for key in BODY_KEYS):
            return
        self._check_embodiment(feedback)
        self._measured = {key: float(feedback[key]) for key in BODY_KEYS}
        if self._synced_to_robot:
            return
        for key in BODY_KEYS + (HEAD_KEYS if self._head else ()):
            self._target[key] = float(feedback[key])
        for side in HAND_SIDES if self.hand_spec is not None else ():
            key = hand_closure_key(side)
            joint_keys = self.hand_spec.joint_keys(side)
            if key in feedback:
                self._set_closure(side, _clip(float(feedback[key]), 0.0, 1.0))
            elif all(k in feedback for k in joint_keys):
                self._set_closure(
                    side, self.hand_spec.q_to_closure(side, [float(feedback[k]) for k in joint_keys])
                )
        self._synced_to_robot = True

    def _pressed(self, buttons: dict[str, bool]) -> set[str]:
        pressed = {name for name, down in buttons.items() if down and not self._buttons.get(name, False)}
        self._buttons = dict(buttons)
        return pressed

    def _engage(self) -> None:
        self.xr.recenter()
        q = [
            self._measured.get(arm_key(name), self._target[arm_key(name)]) for name in self.arm_ik.joint_names
        ]
        self.arm_ik.reset(q)
        self._engaged = True
        logger.info("XR teleop engaged")

    def _step_arms(self, frame: Any, dt: float) -> None:
        solution = self.arm_ik.solve(frame.wrists["left"], frame.wrists["right"])
        max_step = self.config.max_arm_speed_rad_s * dt
        for name, q in zip(self.arm_ik.joint_names, solution.q, strict=True):
            key = arm_key(name)
            self._target[key] = _step_toward(self._target[key], float(q), max_step)

    def _step_head(self, frame: Any, dt: float) -> None:
        pitch = -frame.head_pitch if self.config.invert_head_pitch else frame.head_pitch
        max_step = self.config.head_speed_rad_s * dt
        for key, angle in zip(HEAD_KEYS, (frame.head_yaw, pitch), strict=True):
            lo, hi = HEAD_LIMITS_RAD[key.removesuffix(".q")]
            self._target[key] = _step_toward(self._target[key], _clip(angle, lo, hi), max_step)

    def _step_hands(self, frame: Any, dt: float) -> None:
        max_step = self.config.hand_blend_per_s * dt
        for side in HAND_SIDES:
            closure = self._retarget(frame.hands[side])
            if closure is not None:
                self._set_closure(side, _step_toward(self._target[hand_closure_key(side)], closure, max_step))

    def _stick(self, value: float) -> float:
        return 0.0 if abs(value) < self.config.stick_deadzone else _clip(value, -1.0, 1.0)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.perf_counter()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, _DT_CAP_S)
        self._last_t = now

        frame = self.xr.read()
        pressed = self._pressed(frame.buttons)
        if "b" in pressed and self._engaged:
            self._engaged = False
            logger.warning("XR teleop stopped (B): holding every target")
        elif "a" in pressed:
            if self._engaged:
                self._engaged = False
                logger.info("XR teleop disengaged")
            elif frame.tracking:
                self._engage()
        if self.config.input_mode == "hand" and not self._auto_engaged and frame.tracking:
            self._auto_engaged = True
            self._engage()
        if "right_stick" in pressed:
            self._pending_events.add(TeleopEvents.SUCCESS)
        if "left_stick" in pressed:
            self._pending_events.add(TeleopEvents.RERECORD_EPISODE)

        active = self._engaged and frame.tracking
        if active:
            self._step_arms(frame, dt)
            if self._head:
                self._step_head(frame, dt)
            if self.hand_spec is not None:
                self._step_hands(frame, dt)

        (lx, ly), (rx, ry) = frame.sticks["left"], frame.sticks["right"]
        # Vuer's stick y is negative pushed forward.
        axes = dict(
            zip(
                REMOTE_AXES,
                (self._stick(lx), -self._stick(ly), self._stick(rx), -self._stick(ry)),
                strict=True,
            )
        )
        if not active:
            axes = dict.fromkeys(REMOTE_AXES, 0.0)
        lower = active and frame.buttons.get("x", False)
        raise_ = active and frame.buttons.get("y", False)
        lo, hi = GROOT_BASE_HEIGHT_RANGE
        height = self._target[BASE_HEIGHT_KEY] + GROOT_BASE_HEIGHT_RATE * dt * (raise_ - lower)
        self._target[BASE_HEIGHT_KEY] = _clip(height, lo, hi)

        buttons = dict.fromkeys(REMOTE_BUTTONS, 0.0)
        buttons[_WAIST_RAISE_KEY] = float(raise_)
        buttons[_WAIST_LOWER_KEY] = float(lower)
        nav = nav_from_remote(axes["remote.lx"], axes["remote.ly"], axes["remote.rx"])
        return (
            {key: self._target[key] for key in self.action_features if key in self._target}
            | axes
            | nav
            | buttons
        )

    def get_teleop_events(self) -> dict[str, Any]:
        events, self._pending_events = self._pending_events, set()
        return {
            TeleopEvents.IS_INTERVENTION: self._engaged,
            TeleopEvents.TERMINATE_EPISODE: bool(events),
            TeleopEvents.SUCCESS: TeleopEvents.SUCCESS in events,
            TeleopEvents.RERECORD_EPISODE: TeleopEvents.RERECORD_EPISODE in events,
        }
