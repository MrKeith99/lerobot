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

"""Robot side of the head/hand bridge: ZMQ client, tick <-> radian conversion and calibration.

Serves the motors the configured embodiment has behind the bridge: the pan/tilt head, the
AmazingHand servos, or both. Calibrations live in the robot's calibration file, keyed by motor name.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from collections.abc import Mapping

from lerobot.motors.motors_bus import MotorCalibration

from .end_effectors import (
    AMAZING_HAND_LIMIT_RAD,
    AMAZING_HAND_MIDDLE_POS_DEG,
    HAND_SIDES,
    amazing_hand_motor_names,
)
from .headhand_devices import (
    TICK_RANGE,
    TICKS_PER_RAD,
    clamp_rad,
    default_calibration,
    rad_to_ticks,
    ticks_to_rad,
)
from .headhand_zmq import HeadHandZmqClient
from .heads import HEAD_LIMITS_RAD, HEAD_MOTORS

logger = logging.getLogger(__name__)

# Motor names used before the Unitree-style rename, e.g. in older calibration files.
LEGACY_MOTOR_NAMES: dict[str, str] = {
    "xl330_joint": "kHeadYaw",
    "d455_joint": "kHeadPitch",
    **{
        f"{side}_hand_finger{i}_motor{j}": name
        for side in HAND_SIDES
        for (i, j), name in zip(
            ((i, j) for i in range(1, 5) for j in range(1, 3)), amazing_hand_motor_names(side), strict=True
        )
    },
}


def _clamp_to_model_range(model: str, value: int) -> int:
    tick_min, tick_max = TICK_RANGE[model]
    return max(tick_min, min(value, tick_max))


def _ticks(data: Mapping | None) -> dict[str, int]:
    return dict(data["ticks"]) if data and data.get("ticks") else {}


class HeadHandBridge:
    """Reads and commands `motors` ({name: (ID, model)}) through a `HeadHandServer`, in radians."""

    def __init__(
        self,
        motors: Mapping[str, tuple[int, str]],
        ip: str,
        state_port: int,
        cmd_port: int,
        *,
        timeout_s: float = 5.0,
        stale_warn_s: float = 0.5,
    ) -> None:
        self.motors = dict(motors)
        self.keys: tuple[str, ...] = tuple(f"{name}.q" for name in self.motors)
        self.client = HeadHandZmqClient(ip, state_port, cmd_port)
        self.timeout_s = timeout_s
        self.stale_warn_s = stale_warn_s
        self._ticks: dict[str, int] = {}
        self._stale_logged = False
        self._warned_names: set[str] = set()

    def connect(self) -> None:
        """Connect and wait for the first state message."""
        self.client.connect()
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            ticks = _ticks(self.client.read_latest())
            if ticks:
                self._ticks = ticks
                return
            time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for head/hand state on port {self.client.state_port}")

    def disconnect(self) -> None:
        with contextlib.suppress(Exception):
            self.client.disconnect()

    def is_calibrated(self, calibration: Mapping[str, MotorCalibration]) -> bool:
        return all(name in calibration for name in self.motors)

    def default_calibration(self) -> dict[str, MotorCalibration]:
        return {name: default_calibration(name) for name in self.motors}

    def set_torque(self, enabled: bool) -> None:
        self.client.send(torque=enabled)

    def read(self, calibration: Mapping[str, MotorCalibration]) -> dict[str, float]:
        """Latest positions (rad) of the calibrated motors, keyed `{name}.q`."""
        self._ticks.update(_ticks(self.client.read_latest()))

        age_s = self.client.age_s
        if age_s is not None and age_s > self.stale_warn_s:
            if not self._stale_logged:
                logger.warning(f"Head/hand state is stale ({age_s:.2f}s old)")
                self._stale_logged = True
        else:
            self._stale_logged = False

        return {
            f"{name}.q": ticks_to_rad(model, self._ticks[name], calibration[name])
            for name, (_motor_id, model) in self.motors.items()
            if name in self._ticks and name in calibration
        }

    def write(
        self, targets: Mapping[str, float], calibration: Mapping[str, MotorCalibration]
    ) -> dict[str, float]:
        """Send the `{name}.q` targets (rad) of this bridge's motors; returns the clamped targets sent."""
        goals: dict[str, int] = {}
        sent: dict[str, float] = {}
        for name, (_motor_id, model) in self.motors.items():
            key = f"{name}.q"
            if key not in targets:
                continue
            if name not in calibration:
                if name not in self._warned_names:
                    logger.warning(f"No calibration for {name!r}; skipping in send_action")
                    self._warned_names.add(name)
                continue
            rad = clamp_rad(name, float(targets[key]))
            goals[name] = rad_to_ticks(model, rad, calibration[name])
            sent[key] = rad
        if goals:
            self.client.send(goal_ticks=goals)
        return sent

    def _record_range(self, seconds: float | None = None) -> dict[str, tuple[int, int]]:
        """Sample ticks until Enter is pressed, tracking per-motor min/max."""
        mins: dict[str, int] = {}
        maxs: dict[str, int] = {}
        stop_event = threading.Event()

        def _wait_enter() -> None:
            input()
            stop_event.set()

        waiter = threading.Thread(target=_wait_enter, daemon=True)
        waiter.start()
        deadline = None if seconds is None else time.time() + seconds
        while not stop_event.is_set() and (deadline is None or time.time() < deadline):
            for name, tick in _ticks(self.client.read_latest()).items():
                mins[name] = tick if name not in mins else min(mins[name], tick)
                maxs[name] = tick if name not in maxs else max(maxs[name], tick)
            time.sleep(0.01)
        waiter.join(timeout=0.1)
        return {name: (mins[name], maxs[name]) for name in mins}

    def _calibrate_head(self, head_range, zero_ticks, direction_ticks) -> dict[str, MotorCalibration]:
        calibration: dict[str, MotorCalibration] = {}
        for name, (motor_id, model) in HEAD_MOTORS.items():
            zero = zero_ticks.get(name, default_calibration(name).homing_offset)
            drive_mode = 0
            if name in direction_ticks and name in zero_ticks:
                drive_mode = 0 if direction_ticks[name] >= zero_ticks[name] else 1
            low_rad, high_rad = HEAD_LIMITS_RAD[name]
            sign = -1.0 if drive_mode else 1.0
            tick_low = round(sign * low_rad * TICKS_PER_RAD[model]) + zero
            tick_high = round(sign * high_rad * TICKS_PER_RAD[model]) + zero
            range_min, range_max = min(tick_low, tick_high), max(tick_low, tick_high)
            if name in head_range:
                sample_min, sample_max = head_range[name]
                range_min = max(range_min, sample_min)
                range_max = min(range_max, sample_max)
            calibration[name] = MotorCalibration(
                id=motor_id,
                drive_mode=drive_mode,
                homing_offset=zero,
                range_min=_clamp_to_model_range(model, range_min),
                range_max=_clamp_to_model_range(model, range_max),
            )
        return calibration

    def _hand_calibration(self, name: str, zero: int, sampled: tuple[int, int] | None) -> MotorCalibration:
        motor_id, model = self.motors[name]
        span = round(AMAZING_HAND_LIMIT_RAD * TICKS_PER_RAD[model])
        range_min, range_max = zero - span, zero + span
        if sampled is not None:
            range_min = max(range_min, sampled[0])
            range_max = min(range_max, sampled[1])
        return MotorCalibration(
            id=motor_id,
            drive_mode=0,
            homing_offset=zero,
            range_min=_clamp_to_model_range(model, range_min),
            range_max=_clamp_to_model_range(model, range_max),
        )

    def _calibrate_hands_interactive(self) -> dict[str, MotorCalibration]:
        input("Move both hands to a neutral pose, then press Enter")
        neutral_ticks = _ticks(self.client.read_latest())
        print("Sweep both hands through their full range, then press Enter")
        hand_range = self._record_range()
        return {
            name: self._hand_calibration(
                name, neutral_ticks.get(name, default_calibration(name).homing_offset), hand_range.get(name)
            )
            for side in HAND_SIDES
            for name in amazing_hand_motor_names(side)
        }

    def _calibrate_hands_from_lab_offsets(self) -> dict[str, MotorCalibration]:
        calibration: dict[str, MotorCalibration] = {}
        for side, deg_offsets in AMAZING_HAND_MIDDLE_POS_DEG.items():
            for deg, name in zip(deg_offsets, amazing_hand_motor_names(side), strict=True):
                _motor_id, model = self.motors[name]
                zero = 512 + round(deg * TICKS_PER_RAD[model] * math.pi / 180.0)
                calibration[name] = self._hand_calibration(name, zero, None)
        return calibration

    def calibrate(self) -> dict[str, MotorCalibration]:
        """Interactively calibrate the head and/or hand motors behind this bridge."""
        self.set_torque(False)
        calibration: dict[str, MotorCalibration] = {}

        if all(name in self.motors for name in HEAD_MOTORS):
            input("Center the head (camera forward, level), then press Enter")
            zero_ticks = _ticks(self.client.read_latest())
            input("Turn the head to the robot's LEFT and tilt UP slightly, then press Enter")
            direction_ticks = _ticks(self.client.read_latest())
            print("Sweep pan and tilt to both mechanical stops, then press Enter")
            head_range = self._record_range()
            calibration.update(self._calibrate_head(head_range, zero_ticks, direction_ticks))

        if any(name in self.motors for name in amazing_hand_motor_names("left")):
            hand_mode = input(
                "Press Enter to import the lab middle_pos offsets, or type 'i' for interactive: "
            )
            if hand_mode.strip().lower() == "i":
                calibration.update(self._calibrate_hands_interactive())
            else:
                calibration.update(self._calibrate_hands_from_lab_offsets())

        self.set_torque(True)
        return calibration
