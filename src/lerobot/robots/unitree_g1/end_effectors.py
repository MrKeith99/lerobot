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

"""End effector variants for the Unitree G1: joint keys, open/closed poses and the closure mapping.

- `rubber_hand`: Unitree's stock passive hand (no joints).
- `none`: bare wrist (no joints).
- `dex1`: Unitree Dex1-1 parallel gripper, one position per side, over DDS.
- `dex3`: Unitree Dex3-1 hand, 7 joints per side, over DDS.
- `amazing_hand`: Pollen Robotics AmazingHand, 8 Feetech SCS0009 servos per side, over the head/hand
  ZMQ bridge.

Every hand also has a scalar closure per side (`k{Side}Hand.closure`, 0 = open, 1 = closed), mapped
linearly onto its open/closed poses. Pure stdlib: no hardware/SDK/bus imports here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

END_EFFECTORS: tuple[str, ...] = ("rubber_hand", "none", "dex1", "dex3", "amazing_hand")
HAND_SIDES: tuple[str, ...] = ("left", "right")
# "closure": one value per hand in [0, 1]; "per_motor": every hand joint/servo.
HAND_REPRESENTATIONS: tuple[str, ...] = ("closure", "per_motor")


def hand_closure_key(side: str) -> str:
    """Return the scalar closure key for one hand (0 = open, 1 = closed)."""
    if side not in HAND_SIDES:
        raise ValueError(f"Unknown hand side: {side!r}")
    return f"k{side.capitalize()}Hand.closure"


HAND_CLOSURE_KEYS: tuple[str, ...] = tuple(hand_closure_key(side) for side in HAND_SIDES)


@dataclass(frozen=True)
class HandSpec:
    """Joint names and open/closed poses of one end effector, per side (`left`, `right`)."""

    name: str
    motor_names: dict[str, tuple[str, ...]]
    open_q: dict[str, tuple[float, ...]]
    closed_q: dict[str, tuple[float, ...]]

    def joint_keys(self, side: str | None = None) -> tuple[str, ...]:
        """`.q` keys of one side, or of both sides (left first) when `side` is None."""
        sides = HAND_SIDES if side is None else (side,)
        return tuple(f"{name}.q" for s in sides for name in self.motor_names[s])

    def closure_to_q(self, side: str, closure: float) -> tuple[float, ...]:
        """Joint targets for a closure in [0, 1] (clipped)."""
        c = min(max(float(closure), 0.0), 1.0)
        return tuple(o + (k - o) * c for o, k in zip(self.open_q[side], self.closed_q[side], strict=True))

    def q_to_closure(self, side: str, q: Sequence[float]) -> float:
        """Closure in [0, 1] of joint positions: the mean of each joint's open->closed fraction."""
        open_q, closed_q = self.open_q[side], self.closed_q[side]
        if len(q) != len(open_q):
            raise ValueError(f"Expected {len(open_q)} {self.name} positions, got {len(q)}")
        fractions = [(v - o) / (k - o) for v, o, k in zip(q, open_q, closed_q, strict=True) if k != o]
        return min(max(sum(fractions) / len(fractions), 0.0), 1.0)


# AmazingHand: 4 fingers x 2 servos, named by motor ID (right 1-8, left 11-18), ordered
# finger1 servo1 .. finger4 servo2.
AMAZING_HAND_MOTOR_IDS: dict[str, tuple[int, ...]] = {
    "right": tuple(range(1, 9)),
    "left": tuple(range(11, 19)),
}
AMAZING_HAND_MOTOR_MODEL = "scs0009"
AMAZING_HAND_LIMIT_RAD = math.radians(90.0)
AMAZING_HAND_OPEN_DEG: tuple[float, float] = (-35.0, 35.0)
AMAZING_HAND_CLOSE_DEG: tuple[float, float] = (60.0, -60.0)
AMAZING_HAND_THUMB_CLOSE_DEG: tuple[float, float] = (75.0, -75.0)
AMAZING_HAND_THUMB_FINGER = 4  # verify on hardware
# Lab per-servo zero trims (deg), applied at the tick layer by calibration only.
AMAZING_HAND_MIDDLE_POS_DEG: dict[str, tuple[float, ...]] = {
    "right": (25, -20, 25, -15, 20, -25, 15, -15),
    "left": (20, -10, 33, -35, 30, -10, 20, -8),
}


def amazing_hand_motor_names(side: str) -> tuple[str, ...]:
    """Return the 8 servo names of one AmazingHand."""
    if side not in HAND_SIDES:
        raise ValueError(f"Unknown hand side: {side!r}")
    return tuple(f"k{side.capitalize()}HandMotor{motor_id}" for motor_id in AMAZING_HAND_MOTOR_IDS[side])


AMAZING_HAND_MOTORS: dict[str, tuple[int, str]] = {
    name: (motor_id, AMAZING_HAND_MOTOR_MODEL)
    for side in HAND_SIDES
    for name, motor_id in zip(amazing_hand_motor_names(side), AMAZING_HAND_MOTOR_IDS[side], strict=True)
}


def amazing_hand_pose_deg(closed: bool) -> tuple[float, ...]:
    """The 8 servo targets (deg) of a fully open or closed AmazingHand, excluding the zero trims."""
    pose: list[float] = []
    for finger in range(1, 5):
        if closed and finger == AMAZING_HAND_THUMB_FINGER:
            pose.extend(AMAZING_HAND_THUMB_CLOSE_DEG)
        elif closed:
            pose.extend(AMAZING_HAND_CLOSE_DEG)
        else:
            pose.extend(AMAZING_HAND_OPEN_DEG)
    return tuple(pose)


def _amazing_hand_pose_rad(closed: bool) -> tuple[float, ...]:
    return tuple(math.radians(deg) for deg in amazing_hand_pose_deg(closed))


AMAZING_HAND = HandSpec(
    name="amazing_hand",
    motor_names={side: amazing_hand_motor_names(side) for side in HAND_SIDES},
    open_q=dict.fromkeys(HAND_SIDES, _amazing_hand_pose_rad(closed=False)),
    closed_q=dict.fromkeys(HAND_SIDES, _amazing_hand_pose_rad(closed=True)),
)

# Dex3-1: joint order of the MuJoCo model (thumb, middle, index). The real right hand orders its
# motors thumb, index, middle; the DDS driver handles that.
DEX3_JOINTS: tuple[str, ...] = ("Thumb0", "Thumb1", "Thumb2", "Middle0", "Middle1", "Index0", "Index1")
DEX3 = HandSpec(
    name="dex3",
    motor_names={
        side: tuple(f"k{side.capitalize()}Hand{joint}" for joint in DEX3_JOINTS) for side in HAND_SIDES
    },
    open_q=dict.fromkeys(HAND_SIDES, (0.0,) * len(DEX3_JOINTS)),
    # ~80-90% of each joint's flexion limit; the right hand flexes the other way.
    closed_q={
        "left": (0.0, 0.7, 1.4, -1.3, -1.4, -1.3, -1.4),
        "right": (0.0, -0.7, -1.4, 1.3, 1.4, 1.3, 1.4),
    },
)

# Dex1-1: finger travel in m, as in the MuJoCo model (range -0.023..0.0245). Closing past contact
# makes the grip force; the real gripper's motor units are converted by the DDS driver.
DEX1 = HandSpec(
    name="dex1",
    motor_names={side: (f"k{side.capitalize()}Gripper",) for side in HAND_SIDES},
    open_q=dict.fromkeys(HAND_SIDES, (0.0245,)),
    closed_q=dict.fromkeys(HAND_SIDES, (-0.02,)),
)

HAND_SPECS: dict[str, HandSpec] = {spec.name: spec for spec in (DEX1, DEX3, AMAZING_HAND)}
# Every per-joint hand key of every end effector, e.g. for a teleoperator serving any of them.
ALL_HAND_KEYS: tuple[str, ...] = tuple(key for spec in HAND_SPECS.values() for key in spec.joint_keys())
