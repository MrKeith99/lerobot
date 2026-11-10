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

"""Pan/tilt head (Dynamixel) + AmazingHand (Feetech) motor bus wrapper for the Unitree G1.

Ticks are always read/written raw (`normalize=False`); conversion to/from radians is
done here instead of relying on `MotorsBus` normalization, since the SCS0009 servo span
does not match any of `MotorNormMode`'s built-in modes.
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
import time
from collections.abc import Mapping

from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode

from .end_effectors import AMAZING_HAND_LIMIT_RAD, AMAZING_HAND_MOTORS as HAND_MOTORS
from .hand_force import HandForceLimiter, decode_scs_load
from .heads import HEAD_LIMITS_RAD, HEAD_MOTORS

logger = logging.getLogger(__name__)

MAX_LIMITER_DT_S = 0.1

TICKS_PER_RAD: dict[str, float] = {
    "xl330-m288": 4096 / (2 * math.pi),
    "scs0009": 1024 / math.radians(300.0),  # verify SCS0009 span on hardware
}
TICK_RANGE: dict[str, tuple[int, int]] = {
    "xl330-m288": (
        -1_048_575,
        1_048_575,
    ),  # the head runs in mode 5 (current-based position), which is multi-turn
    "scs0009": (0, 1023),
}
TICKS_PER_TURN = 4096

# Mode 5 (the erasers' setting): multi-turn, EEPROM position limits not applied; Goal_Current (RAM, 1 mA/unit) caps
# the push so a stop is only pressed gently. Current_Limit is never written.
HEAD_OPERATING_MODE = 5
HEAD_GOAL_CURRENT = 75  # starts both joints from standstill (40 did not); 2026-11-10 sweeps
# Hard stops minus 30 ticks, measured 2026-11-10 (yaw 1704..2861; pitch -553..678 worst case over yaw, 0 = front).
HEAD_BOX_TICKS: dict[str, tuple[int, int]] = {"kHeadYaw": (1734, 2831), "kHeadPitch": (-523, 648)}
# Within this many ticks outside the box (pitch rests on its down stop at ~700) a joint still gets torque.
HEAD_BOX_TOLERANCE = 150

DEFAULT_HEAD_PORT = "/dev/ttyCH341USB0"
DEFAULT_HAND_PORT = "/dev/ttyACM0"

# Every motor behind the head/hand bridge, by name: (ID, model).
HEAD_HAND_MOTORS: dict[str, tuple[int, str]] = {**HEAD_MOTORS, **HAND_MOTORS}


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
        return min(AMAZING_HAND_LIMIT_RAD, max(-AMAZING_HAND_LIMIT_RAD, rad))
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
    """Owns the Dynamixel head bus and/or the Feetech AmazingHand bus; a `None` port skips that bus."""

    def __init__(
        self,
        head_port: str | None,
        hand_port: str | None,
        *,
        head_bus_cls=None,
        hand_bus_cls=None,
        hand_force: HandForceLimiter | None = None,
        temp_every: int = 25,
        head_box: Mapping[str, tuple[int, int]] | None = None,
        head_goal_current: int = HEAD_GOAL_CURRENT,
    ) -> None:
        if head_port is None and hand_port is None:
            raise ValueError("HeadHandDevice needs a head port, a hand port or both")
        if head_bus_cls is None and head_port is not None:
            from lerobot.motors.dynamixel.dynamixel import DynamixelMotorsBus

            head_bus_cls = DynamixelMotorsBus
        if hand_bus_cls is None and hand_port is not None:
            from lerobot.motors.feetech.feetech import FeetechMotorsBus

            hand_bus_cls = FeetechMotorsBus

        self.head_bus = (
            None if head_port is None else head_bus_cls(port=head_port, motors=build_head_motors())
        )
        self.hand_bus = (
            None
            if hand_port is None
            else hand_bus_cls(port=hand_port, motors=build_hand_motors(), protocol_version=1)
        )
        motors = {**(HEAD_MOTORS if self.head_bus else {}), **(HAND_MOTORS if self.hand_bus else {})}
        self.models: dict[str, str] = {name: model for name, (_, model) in motors.items()}
        self._lock = threading.Lock()
        self.hand_force = hand_force if self.hand_bus is not None else None
        self.temp_every = max(1, temp_every)
        self._read_cycles = 0
        self._last_write_t: float | None = None
        self._hand_present: dict[str, int] = {}
        self._hand_load: dict[str, int] = {}
        self._hand_temp: dict[str, int] = {}
        self._hand_torque_off: set[str] = set()
        self._torque_on = False
        self._head_box = dict(HEAD_BOX_TICKS if head_box is None else head_box)
        self.head_goal_current = head_goal_current
        self.head_limits: dict[str, tuple[int, int]] = {}
        self.head_faults: dict[str, str] = {}
        # The turn count restarts in 0..4095 at power-up (pitch looking up reads 3096..4095, not -1000..-1).
        self.head_turns: dict[str, int] = {}

    @property
    def _buses(self) -> list:
        return [bus for bus in (self.head_bus, self.hand_bus) if bus is not None]

    def connect(self, handshake: bool = True) -> None:
        with self._lock:
            for bus in self._buses:
                bus.connect(handshake=handshake)

    def disconnect(self, disable_torque: bool = False) -> None:
        with self._lock:
            for bus in self._buses:
                if disable_torque and bus is self.hand_bus:
                    self._hand_torque_off_all()
                    bus.disconnect(disable_torque=False)
                else:
                    bus.disconnect(disable_torque=disable_torque)

    def configure(self) -> None:
        with self._lock:
            if self.head_bus is not None:
                # Torque stays off (`torque_disabled()` would re-enable it at a stale goal); no EEPROM writes on
                # the shared head, so no configure_motors() for it.
                self.head_bus.disable_torque()
                for name in HEAD_MOTORS:
                    self.head_bus.write("Operating_Mode", name, HEAD_OPERATING_MODE, normalize=False)
                    self.head_bus.write("Goal_Current", name, self.head_goal_current, normalize=False)
                self.head_limits = {name: box for name, box in self._head_box.items() if name in HEAD_MOTORS}
                self.head_turns = dict.fromkeys(self.head_limits, 0)
            if self.hand_bus is not None:
                self.hand_bus.configure_motors()

    def _unwrap_head(self, name: str, raw: int) -> int:
        """Raw head tick -> box ticks: the whole-turn shift (the last one first) that lands within tolerance."""
        if name not in self.head_limits:
            return raw
        low, high = self.head_limits[name]
        for shift in (self.head_turns[name], 0, -TICKS_PER_TURN, TICKS_PER_TURN):
            if low - HEAD_BOX_TOLERANCE <= raw + shift <= high + HEAD_BOX_TOLERANCE:
                self.head_turns[name] = shift
                return raw + shift
        return raw + self.head_turns[name]

    def _check_head_range(self) -> dict[str, int]:
        """Raw present ticks of the head motors within tolerance of their box; the others go to `head_faults`."""
        present = self.head_bus.sync_read("Present_Position", normalize=False)
        allowed: dict[str, int] = {}
        for name, (low, high) in self.head_limits.items():
            raw = int(present[name])
            tick = self._unwrap_head(name, raw)
            if low - HEAD_BOX_TOLERANCE <= tick <= high + HEAD_BOX_TOLERANCE:
                allowed[name] = raw
                self.head_faults.pop(name, None)
                continue
            message = (
                f"{name} reads {raw}, more than {HEAD_BOX_TOLERANCE} ticks outside its box {low}..{high}: torque "
                "stays off. Turn it back by hand; it is checked again at the next torque-on."
            )
            if self.head_faults.get(name) is None:
                logger.error(message)
            self.head_faults[name] = message
        return allowed

    @property
    def fault(self) -> str | None:
        """Why some head motors are held without torque, or None."""
        return "; ".join(self.head_faults.values()) or None

    def set_torque(self, enabled: bool) -> None:
        with self._lock:
            self._torque_on = enabled
            for bus in self._buses:
                if enabled and bus is self.head_bus and self.head_limits:
                    allowed = self._check_head_range()
                    if allowed:
                        # hold where it is: a stale goal from before the torque-off would jump
                        bus.sync_write("Goal_Position", allowed, normalize=False)
                        bus.enable_torque(list(allowed))
                elif enabled and bus is self.hand_bus and self._hand_torque_off:
                    allowed = [name for name in HAND_MOTORS if name not in self._hand_torque_off]
                    if allowed:
                        bus.enable_torque(allowed)
                elif enabled:
                    bus.enable_torque()
                elif bus is self.hand_bus:
                    self._hand_torque_off_all()
                else:
                    bus.disable_torque()

    def _hand_torque_off_all(self) -> None:
        """Torque off every hand servo even when some replies carry an error flag (e.g. input voltage) or a hand is
        missing: the bus's `disable_torque` stops at the first error and would leave the rest powered. An
        unacknowledged broadcast goes first, then each servo on its own; errors are logged, never raised."""
        try:
            self.hand_bus.sync_write("Torque_Enable", dict.fromkeys(HAND_MOTORS, 0), normalize=False)
        except Exception as e:  # noqa: BLE001 - the per-servo writes below still run
            logger.warning("Hand torque-off broadcast failed: %s", e)
        for name in HAND_MOTORS:
            try:
                self.hand_bus.disable_torque(name)
            except Exception as e:  # noqa: BLE001 - one bad servo must not keep the others powered
                logger.warning("Hand torque-off %s: %s", name, e)

    def read_ticks(self) -> dict[str, int]:
        with self._lock:
            ticks: dict[str, int] = {}
            if self.head_bus is not None:
                positions = self.head_bus.sync_read("Present_Position", normalize=False)
                ticks.update((name, self._unwrap_head(name, int(value))) for name, value in positions.items())
            if self.hand_bus is not None:
                if self.hand_force is None:
                    for name in HAND_MOTORS:
                        ticks[name] = int(
                            self.hand_bus.read("Present_Position", name, normalize=False, num_retry=1)
                        )
                else:
                    with_temp = self._read_cycles % self.temp_every == 0
                    self._read_cycles += 1
                    ticks.update(self._read_hand_feedback(with_temp))
            return ticks

    def _read_hand_feedback(self, with_temp: bool) -> dict[str, int]:
        for name in HAND_MOTORS:
            self._hand_present[name] = int(
                self.hand_bus.read("Present_Position", name, normalize=False, num_retry=1)
            )
            self._hand_load[name] = decode_scs_load(
                self.hand_bus.read("Present_Load", name, normalize=False, num_retry=1)
            )
            if with_temp:
                self._hand_temp[name] = int(
                    self.hand_bus.read("Present_Temperature", name, normalize=False, num_retry=1)
                )
        return {name: self._hand_present[name] for name in HAND_MOTORS}

    def read_hand_feedback(self) -> dict[str, tuple[int, int, int]]:
        """Read `(position, signed load per mille, temperature C)` for every hand servo."""
        with self._lock:
            if self.hand_bus is None:
                return {}
            self._read_hand_feedback(with_temp=True)
            return {
                name: (self._hand_present[name], self._hand_load[name], self._hand_temp[name])
                for name in HAND_MOTORS
            }

    def write_ticks(self, goals: Mapping[str, int]) -> None:
        head_goals: dict[str, int] = {}
        hand_goals: dict[str, int] = {}
        for name, tick in goals.items():
            if name not in self.models:
                continue
            tick_min, tick_max = TICK_RANGE[self.models[name]]
            clamped = min(tick_max, max(tick_min, int(tick)))
            if name in HEAD_MOTORS:
                if name in self.head_faults:
                    continue  # torque off until it is turned back by hand
                if name in self.head_limits:
                    low, high = self.head_limits[name]
                    clamped = min(high, max(low, clamped)) - self.head_turns[name]  # box ticks -> raw
                head_goals[name] = clamped
            else:
                hand_goals[name] = clamped

        with self._lock:
            if head_goals:
                self.head_bus.sync_write("Goal_Position", head_goals, normalize=False)
            if hand_goals and self.hand_force is not None:
                hand_goals = self._limit_hand_goals(hand_goals)
            if hand_goals:
                self.hand_bus.sync_write("Goal_Position", hand_goals, normalize=False)

    def _limit_hand_goals(self, hand_goals: dict[str, int]) -> dict[str, int]:
        now = time.monotonic()
        dt = 0.0 if self._last_write_t is None else min(now - self._last_write_t, MAX_LIMITER_DT_S)
        self._last_write_t = now
        limited, torque_off, events = self.hand_force.limit(
            hand_goals, self._hand_present, self._hand_load, self._hand_temp, dt
        )
        for event in events:
            if "released" in event:
                logger.info("Hand force: %s", event)
            else:
                logger.warning("Hand force: %s", event)
        for name, off in torque_off.items():
            if off and name not in self._hand_torque_off:
                self._hand_torque_off.add(name)
                self.hand_bus.disable_torque(name)
            elif not off and name in self._hand_torque_off:
                self._hand_torque_off.discard(name)
                if self._torque_on:
                    self.hand_bus.enable_torque(name)
        return {name: tick for name, tick in limited.items() if name not in self._hand_torque_off}

    def ping_all(self) -> dict[str, int | None]:
        results: dict[str, int | None] = {}
        with self._lock:
            for bus, motors in ((self.head_bus, HEAD_MOTORS), (self.hand_bus, HAND_MOTORS)):
                if bus is None:
                    continue
                for name in motors:
                    try:
                        results[name] = bus.ping(name)
                    except Exception:
                        results[name] = None
        return results

    @property
    def is_connected(self) -> bool:
        return all(bus.is_connected for bus in self._buses)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Unitree G1 head/hand device utility")
    parser.add_argument("command", choices=["scan", "read", "loads", "torque-off"])
    parser.add_argument("--head-port", default=DEFAULT_HEAD_PORT)
    parser.add_argument("--hand-port", default=DEFAULT_HAND_PORT)
    parser.add_argument("--no-head", action="store_true", help="Skip the head bus")
    parser.add_argument("--no-hands", action="store_true", help="Skip the hand bus")
    parser.add_argument("--no-handshake", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    device = HeadHandDevice(
        None if args.no_head else args.head_port, None if args.no_hands else args.hand_port
    )
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
        elif args.command == "loads":
            while True:
                feedback = device.read_hand_feedback()
                print(f"{'servo':<20}{'pos':>6}{'load':>7}{'temp_C':>8}")
                for name, (pos, load, temp) in feedback.items():
                    print(f"{name:<20}{pos:>6}{load:>7}{temp:>8}")
                if not args.loop:
                    break
                time.sleep(0.2)
        elif args.command == "torque-off":
            device.set_torque(False)
    finally:
        device.disconnect()


if __name__ == "__main__":
    _cli()
