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

"""Head (Dynamixel) + hands (Feetech) motor bus wrapper for the UnitreeG1Ah robot.

Ticks are always read/written raw (`normalize=False`); conversion to/from radians is
done here instead of relying on `MotorsBus` normalization, since the SCS0009 servo span
does not match any of `MotorNormMode`'s built-in modes.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from collections.abc import Mapping

from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode

from .g1_ah_joints import HAND_LIMIT_RAD, HAND_MOTORS, HEAD_LIMITS_RAD, HEAD_MOTORS

TICKS_PER_RAD: dict[str, float] = {
    "xl330-m288": 4096 / (2 * math.pi),
    "scs0009": 1024 / math.radians(300.0),  # verify SCS0009 span on hardware
}
TICK_RANGE: dict[str, tuple[int, int]] = {
    "xl330-m288": (0, 4095),
    "scs0009": (0, 1023),
}

DEFAULT_HEAD_PORT = "/dev/ttyCH341USB0"
DEFAULT_HAND_PORT = "/dev/ttyACM0"


def ticks_to_rad(model: str, ticks: int, calib: MotorCalibration) -> float:
    """Convert a raw tick count to radians using the motor's calibration."""
    sign = -1.0 if calib.drive_mode else 1.0
    return sign * (ticks - calib.homing_offset) / TICKS_PER_RAD[model]


def rad_to_ticks(model: str, rad: float, calib: MotorCalibration) -> int:
    """Convert radians to a raw tick count, clamped to calibration and model tick range."""
    sign = -1.0 if calib.drive_mode else 1.0
    ticks = round(sign * rad * TICKS_PER_RAD[model]) + calib.homing_offset
    ticks = min(calib.range_max, max(calib.range_min, ticks))
    tick_min, tick_max = TICK_RANGE[model]
    return min(tick_max, max(tick_min, ticks))


def clamp_rad(name: str, rad: float) -> float:
    """Clamp a joint target in radians to its configured limit."""
    if name in HEAD_LIMITS_RAD:
        low, high = HEAD_LIMITS_RAD[name]
        return min(high, max(low, rad))
    if name in HAND_MOTORS:
        return min(HAND_LIMIT_RAD, max(-HAND_LIMIT_RAD, rad))
    raise ValueError(f"Unknown motor name: {name!r}")


def default_calibration(name: str) -> MotorCalibration:
    """Return a bootstrapping calibration for `name` (head: homing 2048, hands: homing 512)."""
    if name in HEAD_MOTORS:
        motor_id, model = HEAD_MOTORS[name]
        tick_min, tick_max = TICK_RANGE[model]
        return MotorCalibration(
            id=motor_id, drive_mode=0, homing_offset=2048, range_min=tick_min, range_max=tick_max
        )
    if name in HAND_MOTORS:
        motor_id, model = HAND_MOTORS[name]
        tick_min, tick_max = TICK_RANGE[model]
        return MotorCalibration(
            id=motor_id, drive_mode=0, homing_offset=512, range_min=tick_min, range_max=tick_max
        )
    raise ValueError(f"Unknown motor name: {name!r}")


def build_head_motors() -> dict[str, Motor]:
    """Build the `Motor` mapping for the head bus from `HEAD_MOTORS`."""
    return {
        name: Motor(id=motor_id, model=model, norm_mode=MotorNormMode.RANGE_M100_100)
        for name, (motor_id, model) in HEAD_MOTORS.items()
    }


def build_hand_motors() -> dict[str, Motor]:
    """Build the `Motor` mapping for the hand bus from `HAND_MOTORS`."""
    return {
        name: Motor(id=motor_id, model=model, norm_mode=MotorNormMode.RANGE_M100_100)
        for name, (motor_id, model) in HAND_MOTORS.items()
    }


class HeadHandDevice:
    """Owns the Dynamixel head bus and the Feetech hand bus for the UnitreeG1Ah robot."""

    def __init__(
        self,
        head_port: str,
        hand_port: str,
        *,
        head_bus_cls=None,
        hand_bus_cls=None,
    ) -> None:
        if head_bus_cls is None:
            from lerobot.motors.dynamixel.dynamixel import DynamixelMotorsBus

            head_bus_cls = DynamixelMotorsBus
        if hand_bus_cls is None:
            from lerobot.motors.feetech.feetech import FeetechMotorsBus

            hand_bus_cls = FeetechMotorsBus

        self.head_bus = head_bus_cls(port=head_port, motors=build_head_motors())
        self.hand_bus = hand_bus_cls(port=hand_port, motors=build_hand_motors(), protocol_version=1)
        self.models: dict[str, str] = {
            name: model for name, (_, model) in {**HEAD_MOTORS, **HAND_MOTORS}.items()
        }
        self._lock = threading.Lock()

    def connect(self, handshake: bool = True) -> None:
        with self._lock:
            self.head_bus.connect(handshake=handshake)
            self.hand_bus.connect(handshake=handshake)

    def disconnect(self, disable_torque: bool = False) -> None:
        with self._lock:
            self.head_bus.disconnect(disable_torque=disable_torque)
            self.hand_bus.disconnect(disable_torque=disable_torque)

    def configure(self) -> None:
        with self._lock:
            with self.head_bus.torque_disabled():
                for name in HEAD_MOTORS:
                    self.head_bus.write("Operating_Mode", name, 3, normalize=False)
            self.head_bus.configure_motors()
            self.hand_bus.configure_motors()

    def set_torque(self, enabled: bool) -> None:
        with self._lock:
            if enabled:
                self.head_bus.enable_torque()
                self.hand_bus.enable_torque()
            else:
                self.head_bus.disable_torque()
                self.hand_bus.disable_torque()

    def read_ticks(self) -> dict[str, int]:
        with self._lock:
            ticks: dict[str, int] = {
                name: int(value)
                for name, value in self.head_bus.sync_read("Present_Position", normalize=False).items()
            }
            for name in HAND_MOTORS:
                ticks[name] = int(self.hand_bus.read("Present_Position", name, normalize=False, num_retry=1))
            return ticks

    def write_ticks(self, goals: Mapping[str, int]) -> None:
        head_goals: dict[str, int] = {}
        hand_goals: dict[str, int] = {}
        for name, tick in goals.items():
            model = self.models[name]
            tick_min, tick_max = TICK_RANGE[model]
            clamped = min(tick_max, max(tick_min, int(tick)))
            if name in HEAD_MOTORS:
                head_goals[name] = clamped
            elif name in HAND_MOTORS:
                hand_goals[name] = clamped

        with self._lock:
            if head_goals:
                self.head_bus.sync_write("Goal_Position", head_goals, normalize=False)
            if hand_goals:
                self.hand_bus.sync_write("Goal_Position", hand_goals, normalize=False)

    def ping_all(self) -> dict[str, int | None]:
        results: dict[str, int | None] = {}
        with self._lock:
            for name in HEAD_MOTORS:
                try:
                    results[name] = self.head_bus.ping(name)
                except Exception:
                    results[name] = None
            for name in HAND_MOTORS:
                try:
                    results[name] = self.hand_bus.ping(name)
                except Exception:
                    results[name] = None
        return results

    @property
    def is_connected(self) -> bool:
        return bool(self.head_bus.is_connected and self.hand_bus.is_connected)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="UnitreeG1Ah head/hand device utility")
    parser.add_argument("command", choices=["scan", "read", "torque-off"])
    parser.add_argument("--head-port", default=DEFAULT_HEAD_PORT)
    parser.add_argument("--hand-port", default=DEFAULT_HAND_PORT)
    parser.add_argument("--no-handshake", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    device = HeadHandDevice(args.head_port, args.hand_port)
    device.connect(handshake=not args.no_handshake)
    try:
        if args.command == "scan":
            print(device.ping_all())
        elif args.command == "read":
            while True:
                print(device.read_ticks())
                if not args.loop:
                    break
                time.sleep(0.1)
        elif args.command == "torque-off":
            device.set_torque(False)
    finally:
        device.disconnect()


if __name__ == "__main__":
    _cli()
