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

"""Per-servo force limiting for the AmazingHand (Feetech SCS0009), hardware-free.

The SCS0009 has no RAM torque-limit register, so force is bounded indirectly: the goal is kept
close to the present position (a P-controller's force grows with the error) and a servo that
stays loaded is backed off. The limiter only sees ticks, so it needs no closing direction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

SCS_LOAD_MAGNITUDE_MASK = 0x3FF
SCS_LOAD_DIRECTION_BIT = 10
TEMP_HYSTERESIS_C = 5


def decode_scs_load(raw: int) -> int:
    """Decode a raw SCS `Present_Load` word to signed per mille of max torque."""
    magnitude = int(raw) & SCS_LOAD_MAGNITUDE_MASK
    return -magnitude if (int(raw) >> SCS_LOAD_DIRECTION_BIT) & 1 else magnitude


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


@dataclass
class HandForceLimiter:
    max_err_ticks: int = 30
    load_limit: int = 500
    overload_s: float = 0.3
    backoff_ticks: int = 15
    temp_limit_c: int = 65
    _overload_time: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _latched: dict[str, tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    _hot: set[str] = field(default_factory=set, init=False, repr=False)

    def reset(self) -> None:
        self._overload_time.clear()
        self._latched.clear()
        self._hot.clear()

    def limit(
        self,
        goals: dict[str, int],
        present: dict[str, int],
        load: dict[str, int],
        temp: dict[str, int],
        dt: float,
    ) -> tuple[dict[str, int], dict[str, bool], list[str]]:
        """Return `(limited_goals, torque_off, events)` for the servos in `goals`."""
        out: dict[str, int] = {}
        torque_off: dict[str, bool] = {}
        events: list[str] = []
        for name, goal in goals.items():
            pos = present.get(name)
            if pos is None:
                out[name] = goal
                torque_off[name] = name in self._hot
                continue
            torque_off[name] = self._update_temperature(name, temp.get(name), events)
            if torque_off[name]:
                out[name] = pos
                continue
            out[name] = self._limit_one(name, goal, pos, load.get(name), dt, events)
        return out, torque_off, events

    def _update_temperature(self, name: str, temp: int | None, events: list[str]) -> bool:
        if temp is not None:
            if name not in self._hot and temp >= self.temp_limit_c:
                self._hot.add(name)
                events.append(f"{name}: {temp} C >= {self.temp_limit_c} C, torque off")
            elif name in self._hot and temp < self.temp_limit_c - TEMP_HYSTERESIS_C:
                self._hot.discard(name)
                events.append(f"{name}: {temp} C, cooled down, torque allowed again")
        return name in self._hot

    def _limit_one(
        self, name: str, goal: int, pos: int, load: int | None, dt: float, events: list[str]
    ) -> int:
        err = goal - pos
        latched = self._latched.get(name)
        if latched is not None:
            latch_pos, push = latched
            release = -push
            if (goal - latch_pos) * release >= self.backoff_ticks or 0 <= (
                err * release
            ) <= self.max_err_ticks:
                del self._latched[name]
                self._overload_time[name] = 0.0
                events.append(f"{name}: released, goal opened")
            else:
                return self._cap(pos, latch_pos - push * self.backoff_ticks)

        if load is not None and abs(load) > self.load_limit and err != 0:
            self._overload_time[name] = self._overload_time.get(name, 0.0) + dt
            if self._overload_time[name] >= self.overload_s:
                push = _sign(err)
                self._latched[name] = (pos, push)
                self._overload_time[name] = 0.0
                events.append(
                    f"{name}: load {load} > {self.load_limit}, backing off {self.backoff_ticks} ticks"
                )
                return self._cap(pos, pos - push * self.backoff_ticks)
        else:
            self._overload_time[name] = 0.0
        return self._cap(pos, goal)

    def _cap(self, pos: int, goal: int) -> int:
        return pos + max(-self.max_err_ticks, min(self.max_err_ticks, goal - pos))


def add_hand_force_args(parser: argparse.ArgumentParser) -> None:
    defaults = HandForceLimiter()
    parser.add_argument(
        "--hand-force-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Per-servo force limiting on the AmazingHand bus (default: on)",
    )
    parser.add_argument("--hand-max-err-ticks", type=int, default=defaults.max_err_ticks)
    parser.add_argument("--hand-load-limit", type=int, default=defaults.load_limit)
    parser.add_argument("--hand-overload-s", type=float, default=defaults.overload_s)
    parser.add_argument("--hand-backoff-ticks", type=int, default=defaults.backoff_ticks)
    parser.add_argument("--hand-temp-limit", type=int, default=defaults.temp_limit_c)


def hand_force_from_args(args: argparse.Namespace) -> HandForceLimiter | None:
    if not args.hand_force_limit or getattr(args, "no_hands", False):
        return None
    return HandForceLimiter(
        max_err_ticks=args.hand_max_err_ticks,
        load_limit=args.hand_load_limit,
        overload_s=args.hand_overload_s,
        backoff_ticks=args.hand_backoff_ticks,
        temp_limit_c=args.hand_temp_limit,
    )
