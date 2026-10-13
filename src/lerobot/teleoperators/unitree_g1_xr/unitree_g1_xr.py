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
measured arm pose; disengaging also ends the episode as success), triggers close the hands, the left stick walks, the right stick's x turns (both off with `walk=False`: kNav* and the
walking remote axes stay 0, the action keys are unchanged), X/Y held lower/raise the base, and the
right/left stick clicks end the episode as success/rerecord. B (twice within `delete_confirm_s`) deletes:
the running episode (discarded like a rerecord) while recording, otherwise the last episode saved in this
session. Hand tracking has no buttons: it engages once, when tracking first starts. Targets hold while
tracking is lost.

`lerobot-record` reports its state through `on_record_status()`; with `status_overlay` the headset image gets
a banner with it (ready / REC time / reset / saving / delete prompt / result of the last action).
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
    LOCOMOTION_TOGGLE_KEY,
    NAV_KEYS,
    REMOTE_AXES,
    REMOTE_BUTTONS,
    REMOTE_KEYS,
    ROBOT_TYPE_FEEDBACK_KEY,
    SIM_RESET_KEY,
    nav_from_remote,
)
from lerobot.robots.unitree_g1.heads import DEFAULT_HEAD_Q, HEAD_KEYS, HEAD_LIMITS_RAD
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_unitree_g1_xr import UnitreeG1XRTeleopConfig
from .status_overlay import GREEN, ORANGE, RED, YELLOW, Line, draw_status

logger = logging.getLogger(__name__)

_DT_CAP_S = 0.1
_CAUGHT_UP_RAD = 0.05
_RESET_RESYNC_S = 1.0
# GrootLocomotionController's waist raise/lower slots, as in the gamepad teleop.
_WAIST_RAISE_KEY = "remote.button.0"
_WAIST_LOWER_KEY = "remote.button.4"
_NOTICE_S = 3.0
# `result["kind"]` of on_record_status -> headset text, color.
_RESULTS = {"saved": ("SAVED", GREEN), "discarded": ("DISCARDED", ORANGE), "deleted": ("DELETED", ORANGE)}


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
        self._catching_up = False
        self._auto_engaged = False
        self._robot_type_checked = False
        self._synced_to_robot = False
        self._warned_no_camera = False
        self._measured: dict[str, float] = {}
        self._buttons: dict[str, bool] = {}
        self._pending_events: set[str] = set()
        self._last_t: float | None = None
        self._grips_held_s = 0.0
        self._band_toggle_fired = False
        self._resync_s = 0.0
        self._reset_pending = False
        self._status: dict[str, Any] = {}
        self._notice: tuple[str, tuple[int, int, int], float] | None = None
        self._confirm: tuple[tuple, float] | None = None
        self._last_image: Any = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        keys = BODY_KEYS + (HEAD_KEYS if self._head else ())
        if self.hand_spec is not None:
            keys += HAND_CLOSURE_KEYS + self.hand_spec.joint_keys()
        return dict.fromkeys(
            keys + REMOTE_KEYS + NAV_KEYS + (BASE_HEIGHT_KEY, LOCOMOTION_TOGGLE_KEY, SIM_RESET_KEY), float
        )

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

    def ready_to_record(self) -> bool:
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
                image_shape=cfg.display_image_shape,
                display_fps=cfg.display_fps,
                cert_file=cfg.cert_file,
                key_file=cfg.key_file,
                ego_scale=cfg.display_ego_scale,
            )
        self._last_t = None
        print(f"Unitree G1 XR teleop ({cfg.body}, {cfg.end_effector}, {cfg.head_mount}, {cfg.head_sensor}):")
        print("  Headset browser: https://<this host>:8012 (or https://vuer.ai?ws=wss://<this host>:8012)")
        print(
            "  A: engage / disengage (face the robot's forward direction first)"
            + ("; disengaging ends the episode" if cfg.end_episode_on_disengage else "")
        )
        if cfg.walk:
            print(
                "  Triggers: close hands   Left stick: walk   Right stick x: turn   X / Y: lower / raise base"
            )
        else:
            print("  Triggers: close hands   X / Y: lower / raise base")
            print("  Walking disabled (--teleop.walk=false): the sticks do not walk or turn the robot")
        print("  Right / left stick click: end episode success / rerecord")
        print(
            f"  {cfg.delete_button.upper()} twice (within {cfg.delete_confirm_s:g} s): delete the running "
            "episode, or when not recording the last one saved this session"
        )
        print(
            f"  Both grips {cfg.band_toggle_hold_s:g} s (disengaged): start / stop locomotion "
            "(sim: release / attach the band)"
        )
        if cfg.display_mode == "pass-through":
            print("  Headset shows pass-through (--teleop.display_mode=ego|immersive for the head camera)")
        else:
            print(
                f"  Headset shows {cfg.display_camera!r} ({cfg.display_mode}); Quest 'Pass-through' button: room view"
            )

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

    def _render_camera(self, feedback: dict[str, Any]) -> None:
        cfg = self.config
        if cfg.display_mode == "pass-through" or self.xr is None:
            return
        image = feedback.get(cfg.display_camera)
        if image is None:
            if not self._warned_no_camera:
                self._warned_no_camera = True
                present = sorted(k for k, v in feedback.items() if getattr(v, "ndim", 0) == 3)
                logger.warning(
                    f"No {cfg.display_camera!r} image in the robot observation (images: {present}); the "
                    f"headset shows no camera view. Add the camera with --robot.cameras or set "
                    f"--teleop.display_camera."
                )
            return
        self._last_image = image
        self._show(image)

    def _show(self, image: Any) -> None:
        if self.config.status_overlay and self._has_overlay():
            image = draw_status(image, self._overlay_lines(time.perf_counter()))
        try:
            self.xr.render(image)
        except Exception:
            if not self._warned_no_camera:
                self._warned_no_camera = True
                logger.warning(f"Could not show {self.config.display_camera!r} in the headset", exc_info=True)

    def _has_overlay(self) -> bool:
        return bool(self._status) or self._notice is not None

    def on_record_status(self, status: dict[str, Any]) -> None:
        """State from `lerobot-record`: `state` (idle / recording / reset / saving), `episode` (index of the
        running or next one), `kept` / `target` (episodes), `deletable` (index of the last episode saved this
        session, deletable with B, or None), `elapsed_s` / `episode_time_s` and `waiting_for` (what the
        idle wait needs: "robot" and / or "teleop"; [] when nothing) and `result` (`{"kind": "saved" |
        "discarded" | "deleted", "episode": N}`). Keys are merged into the previous status."""
        status = dict(status)
        result = status.pop("result", None)
        changed = any(self._status.get(k) != v for k, v in status.items() if k != "elapsed_s")
        self._status.update(status)
        if result is not None:
            text, color = _RESULTS[result["kind"]]
            self._notice = (f"{text} ep {result['episode']}", color, time.perf_counter() + _NOTICE_S)
            changed = True
        # save_episode blocks the loop: show the new state at once on the last image.
        if (
            changed
            and self._last_image is not None
            and self.xr is not None
            and self.config.display_mode != "pass-through"
        ):
            self._show(self._last_image)

    def _delete_target(self) -> tuple | None:
        """What B would delete now: ("running",), ("last", episode index) or None."""
        state = self._status.get("state")
        if state == "recording":
            return ("running",)
        deletable = self._status.get("deletable")
        if state == "saving" or deletable is None:
            return None
        return ("last", deletable)

    def _idle_lines(self) -> list[Line]:
        """Idle banner: what `lerobot-record` waits for (`waiting_for`: "robot" / "teleop"), else READY."""
        waiting_for = self._status.get("waiting_for") or []
        if "robot" not in waiting_for:
            return [Line("READY  A: start", GREEN)]
        # The band / locomotion hold gesture only works while disengaged.
        if self._engaged:
            return [
                Line("WAITING", YELLOW),
                Line("A: disengage, then", YELLOW),
                Line("hold both grips 1 s", YELLOW),
            ]
        return [
            Line("WAITING", YELLOW),
            Line("Hold both grips 1 s: release band", YELLOW),
            Line("then A: start"),
        ]

    def _overlay_lines(self, now: float) -> list[Line]:
        status = self._status
        state = status.get("state")
        episode = status.get("episode")
        kept = f"kept {status.get('kept', 0)} / {status.get('target', '?')}"
        if state == "recording":
            elapsed, total = status.get("elapsed_s", 0.0), status.get("episode_time_s")
            clock = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
            if total is not None:
                clock += f" / {int(total) // 60:02d}:{int(total) % 60:02d}"
            base = [Line(f"REC ep {episode}  {clock}", RED, dot=True), Line(f"B: discard  {kept}")]
        elif state == "reset":
            base = [Line("RESET", YELLOW)]
            if (deletable := status.get("deletable")) is not None:
                base.append(Line(f"B: delete last (ep {deletable})  {kept}"))
        elif state == "saving":
            base = [Line(f"SAVING ep {episode}...", YELLOW)]
        else:
            base = self._idle_lines()
            if (deletable := status.get("deletable")) is not None:
                base.append(Line(f"B: delete last (ep {deletable})"))
            base.append(Line(kept))
        if self._confirm is not None and self._confirm[0] == self._delete_target() and now < self._confirm[1]:
            what = "running episode" if self._confirm[0][0] == "running" else f"ep {self._confirm[0][1]}"
            return [
                Line(f"Delete {what}?", YELLOW),
                Line(
                    f"press {self.config.delete_button.upper()} again  {self._confirm[1] - now:.1f}s", YELLOW
                ),
            ]
        if self._notice is not None and now < self._notice[2]:
            return [Line(self._notice[0], self._notice[1]), *base[:1]]
        return base

    def _on_delete_pressed(self, now: float) -> None:
        target = self._delete_target()
        if target is None:
            self._confirm = None
            if self._status.get("state") != "saving":
                self._notice = ("nothing to delete", ORANGE, now + _NOTICE_S)
            return
        if self._confirm is not None and self._confirm[0] == target and now < self._confirm[1]:
            self._confirm = None
            self._pending_events.add(
                TeleopEvents.RERECORD_EPISODE if target[0] == "running" else TeleopEvents.DELETE_LAST_EPISODE
            )
            logger.info("XR teleop: delete confirmed (%s)", target[0])
        else:
            self._confirm = (target, now + self.config.delete_confirm_s)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Check the embodiment, track the measured arm pose and, on the first full observation, hold every
        target at the robot's pose."""
        self._render_camera(feedback)
        if not all(key in feedback for key in BODY_KEYS):
            return
        self._check_embodiment(feedback)
        self._measured = {key: float(feedback[key]) for key in BODY_KEYS}
        if self._synced_to_robot and self._resync_s <= 0.0:
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
        self._catching_up = True
        self._resync_s = 0.0
        logger.info("XR teleop engaged")

    def _step_arms(self, frame: Any, dt: float) -> None:
        solution = self.arm_ik.solve(frame.wrists["left"], frame.wrists["right"])
        # Right after engaging the operator's pose may be far from the robot's: approach it slowly.
        speed = self.config.engage_arm_speed_rad_s if self._catching_up else self.config.max_arm_speed_rad_s
        max_step = speed * dt
        gap = 0.0
        for name, q in zip(self.arm_ik.joint_names, solution.q, strict=True):
            key = arm_key(name)
            self._target[key] = _step_toward(self._target[key], float(q), max_step)
            gap = max(gap, abs(self._target[key] - float(q)))
        if gap < _CAUGHT_UP_RAD:
            self._catching_up = False

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

    def _band_toggle(self, frame: Any, dt: float) -> float:
        both = frame.buttons.get("left_grip", False) and frame.buttons.get("right_grip", False)
        if self._engaged or not both:
            self._grips_held_s = 0.0
            self._band_toggle_fired = False
            return 0.0
        self._grips_held_s += dt
        if self._band_toggle_fired or self._grips_held_s < self.config.band_toggle_hold_s:
            return 0.0
        self._band_toggle_fired = True
        logger.info("XR teleop: both grips held, toggling the sim elastic band")
        return 1.0

    def _fire_sim_reset(self) -> float:
        self._resync_s = _RESET_RESYNC_S
        return 1.0

    def on_episode_end(self) -> None:
        if not self.config.reset_sim_on_episode_end:
            return
        self._engaged = False
        self._catching_up = False
        self._grips_held_s = 0.0
        self._band_toggle_fired = False
        self._reset_pending = True
        logger.info("Episode ended: disengaged, resetting the sim")

    def _sim_reset(self) -> float:
        if not self._reset_pending:
            return 0.0
        self._reset_pending = False
        return self._fire_sim_reset()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.perf_counter()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, _DT_CAP_S)
        self._last_t = now

        frame = self.xr.read()
        pressed = self._pressed(frame.buttons)
        self._resync_s = max(self._resync_s - dt, 0.0)
        if "a" in pressed:
            if self._engaged:
                self._engaged = False
                logger.info("XR teleop disengaged")
                if self.config.end_episode_on_disengage:
                    self._pending_events.add(TeleopEvents.SUCCESS)
            elif frame.tracking:
                self._engage()
        if self.config.input_mode == "hand" and not self._auto_engaged and frame.tracking:
            self._auto_engaged = True
            self._engage()
        if "right_stick" in pressed:
            self._pending_events.add(TeleopEvents.SUCCESS)
        if "left_stick" in pressed:
            self._pending_events.add(TeleopEvents.RERECORD_EPISODE)
        if self.config.delete_button in pressed:
            self._on_delete_pressed(now)

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
        elif not self.config.walk:
            axes |= dict.fromkeys(("remote.lx", "remote.ly", "remote.rx"), 0.0)
        lower = active and frame.buttons.get("x", False)
        raise_ = active and frame.buttons.get("y", False)
        lo, hi = GROOT_BASE_HEIGHT_RANGE
        height = self._target[BASE_HEIGHT_KEY] + GROOT_BASE_HEIGHT_RATE * dt * (raise_ - lower)
        self._target[BASE_HEIGHT_KEY] = _clip(height, lo, hi)

        buttons = dict.fromkeys(REMOTE_BUTTONS, 0.0)
        buttons[_WAIST_RAISE_KEY] = float(raise_)
        buttons[_WAIST_LOWER_KEY] = float(lower)
        nav = nav_from_remote(axes["remote.lx"], axes["remote.ly"], axes["remote.rx"])
        if not self.config.walk:
            nav = dict.fromkeys(nav, 0.0)  # no -0.0 from the sign flips
        band_toggle = {
            LOCOMOTION_TOGGLE_KEY: self._band_toggle(frame, dt),
            SIM_RESET_KEY: self._sim_reset(),
        }
        return (
            {key: self._target[key] for key in self.action_features if key in self._target}
            | axes
            | nav
            | buttons
            | band_toggle
        )

    def get_teleop_events(self) -> dict[str, Any]:
        events, self._pending_events = self._pending_events, set()
        return {
            TeleopEvents.IS_INTERVENTION: self._engaged,
            TeleopEvents.TERMINATE_EPISODE: bool(events - {TeleopEvents.DELETE_LAST_EPISODE}),
            TeleopEvents.DELETE_LAST_EPISODE: TeleopEvents.DELETE_LAST_EPISODE in events,
            TeleopEvents.SUCCESS: TeleopEvents.SUCCESS in events,
            TeleopEvents.RERECORD_EPISODE: TeleopEvents.RERECORD_EPISODE in events,
        }
