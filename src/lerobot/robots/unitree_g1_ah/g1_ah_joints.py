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

"""Single source of truth for the UnitreeG1Ah joint/motor layout.

Covers the Unitree G1 23dof body (a 29dof SDK with 6 unused slots on this hardware
revision), the 2-DoF Dynamixel pan/tilt head, and the two 8-servo AmazingHand hands.
Pure stdlib (+ typing only from ``g1_utils``): no hardware/SDK/bus imports here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lerobot.robots.unitree_g1.g1_utils import REMOTE_AXES, G1_29_JointArmIndex, G1_29_JointIndex

ROBOT_TYPE_BASE = "unitree_g1_23dof_ah8_d455_2dof"
ROBOT_TYPE_REV_1_0 = ROBOT_TYPE_BASE + "_rev_1_0"
REVISIONS: tuple[str, ...] = ("base", "rev_1_0")
ROBOT_TYPE_BY_REVISION: dict[str, str] = {"base": ROBOT_TYPE_BASE, "rev_1_0": ROBOT_TYPE_REV_1_0}
MODE_MACHINE_BY_REVISION: dict[str, int] = {"base": 1, "rev_1_0": 4}
HIP_ROLL_GEAR_BY_REVISION: dict[str, float] = {"base": 14.5, "rev_1_0": 22.5}

G1_23_INVALID_SDK_SLOTS: tuple[int, ...] = (13, 14, 20, 21, 27, 28)

G1_23_BODY_JOINTS: tuple[G1_29_JointIndex, ...] = tuple(
    joint for joint in G1_29_JointIndex if joint not in G1_23_INVALID_SDK_SLOTS
)
G1_23_ARM_JOINTS: tuple[G1_29_JointArmIndex, ...] = tuple(
    joint for joint in G1_29_JointArmIndex if joint not in G1_23_INVALID_SDK_SLOTS
)
G1_23_LEG_SLOTS: tuple[int, ...] = tuple(range(12))

SDK23_TO_29: tuple[int, ...] = tuple(int(joint) for joint in G1_23_BODY_JOINTS)
SDK29_TO_23: dict[int, int] = {sdk29: sdk23 for sdk23, sdk29 in enumerate(SDK23_TO_29)}


def body23_from_sdk29(vec: Sequence[float]) -> list[float]:
    """Project a 29-length SDK vector down to the 23 valid slots on this hardware."""
    if len(vec) != 29:
        raise ValueError(f"Expected a length-29 vector, got length {len(vec)}")
    return [vec[sdk29] for sdk29 in SDK23_TO_29]


def sdk29_from_body23(vec: Sequence[float], fill: float = 0.0) -> list[float]:
    """Expand a 23-length body vector to the full 29-length SDK layout, padding invalid slots."""
    if len(vec) != 23:
        raise ValueError(f"Expected a length-23 vector, got length {len(vec)}")
    out = [fill] * 29
    for sdk23, sdk29 in enumerate(SDK23_TO_29):
        out[sdk29] = vec[sdk23]
    return out


HEAD_MOTORS: dict[str, tuple[int, str]] = {
    "xl330_joint": (1, "xl330-m288"),
    "d455_joint": (2, "xl330-m288"),
}

HAND_SIDES: tuple[str, ...] = ("left", "right")
HAND_FINGERS: tuple[str, ...] = ("index", "middle", "ring", "thumb")
HAND_MOTOR_IDS: dict[str, tuple[int, ...]] = {
    "right": tuple(range(1, 9)),
    "left": tuple(range(11, 19)),
}
HAND_MOTOR_MODEL = "scs0009"


def hand_motor_names(side: str) -> tuple[str, ...]:
    """Return the 8 motor names for one hand, ordered finger1_motor1..finger4_motor2."""
    return tuple(f"{side}_hand_finger{i}_motor{j}" for i in range(1, 5) for j in range(1, 3))


HAND_MOTORS: dict[str, tuple[int, str]] = {
    name: (motor_id, HAND_MOTOR_MODEL)
    for side in HAND_SIDES
    for name, motor_id in zip(hand_motor_names(side), HAND_MOTOR_IDS[side], strict=True)
}
HEAD_HAND_MOTORS: dict[str, tuple[int, str]] = {**HEAD_MOTORS, **HAND_MOTORS}

BODY_KEYS: tuple[str, ...] = tuple(f"{joint.name}.q" for joint in G1_23_BODY_JOINTS)
ARM_KEYS: tuple[str, ...] = tuple(f"{joint.name}.q" for joint in G1_23_ARM_JOINTS)
HEAD_KEYS: tuple[str, ...] = ("xl330_joint.q", "d455_joint.q")
LEFT_HAND_KEYS: tuple[str, ...] = tuple(f"{name}.q" for name in hand_motor_names("left"))
RIGHT_HAND_KEYS: tuple[str, ...] = tuple(f"{name}.q" for name in hand_motor_names("right"))
HAND_KEYS: tuple[str, ...] = LEFT_HAND_KEYS + RIGHT_HAND_KEYS
ALL_ACTION_KEYS: tuple[str, ...] = BODY_KEYS + HEAD_KEYS + LEFT_HAND_KEYS + RIGHT_HAND_KEYS
ARM_MODE_ACTION_KEYS: tuple[str, ...] = ARM_KEYS + HEAD_KEYS + HAND_KEYS + REMOTE_AXES
TELEOP_ACTION_KEYS: tuple[str, ...] = ALL_ACTION_KEYS + REMOTE_AXES
INVALID_BODY_KEYS: tuple[str, ...] = tuple(
    f"{joint.name}.q" for joint in G1_29_JointIndex if joint in G1_23_INVALID_SDK_SLOTS
)
HEAD_HAND_KEYS: tuple[str, ...] = HEAD_KEYS + HAND_KEYS


def key_to_motor_name(key: str) -> str:
    """Strip the trailing '.q' suffix from an action/observation key."""
    return key.removesuffix(".q")


def motor_name_to_key(name: str) -> str:
    """Add the '.q' suffix to a bare motor name."""
    return f"{name}.q"


@dataclass(frozen=True)
class Slices:
    body: slice
    head: slice
    left_hand: slice
    right_hand: slice


def _contiguous_slice(keys: Sequence[str], members: Sequence[str], group_name: str) -> slice:
    indices = [i for i, key in enumerate(keys) if key in members]
    if not indices:
        raise ValueError(f"No keys found for group {group_name!r}")
    start, stop = indices[0], indices[-1] + 1
    if indices != list(range(start, stop)):
        raise ValueError(f"Keys for group {group_name!r} are not contiguous in {keys!r}")
    return slice(start, stop)


def slices_for(keys: Sequence[str]) -> Slices:
    """Compute contiguous body/head/left_hand/right_hand slices within `keys`."""
    return Slices(
        body=_contiguous_slice(keys, BODY_KEYS, "body"),
        head=_contiguous_slice(keys, HEAD_KEYS, "head"),
        left_hand=_contiguous_slice(keys, LEFT_HAND_KEYS, "left_hand"),
        right_hand=_contiguous_slice(keys, RIGHT_HAND_KEYS, "right_hand"),
    )


def action_to_vector(action: Mapping[str, float], keys: Sequence[str]) -> list[float]:
    """Pack an action mapping into a vector ordered by `keys`."""
    missing = [key for key in keys if key not in action]
    if missing:
        raise KeyError(f"Missing keys in action: {missing}")
    return [action[key] for key in keys]


def vector_to_action(vec: Sequence[float], keys: Sequence[str]) -> dict[str, float]:
    """Unpack a vector into an action mapping keyed by `keys`."""
    if len(vec) != len(keys):
        raise ValueError(f"Expected a length-{len(keys)} vector, got length {len(vec)}")
    return dict(zip(keys, vec, strict=True))


HEAD_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "xl330_joint": (-0.7, 0.7),
    "d455_joint": (-1.57, 0.8),
}
HAND_LIMIT_RAD = math.radians(90.0)
HAND_OPEN_DEG: tuple[float, float] = (-35.0, 35.0)
HAND_CLOSE_DEG: tuple[float, float] = (60.0, -60.0)
THUMB_CLOSE_DEG: tuple[float, float] = (75.0, -75.0)
THUMB_FINGER_INDEX = 4  # verify on hardware

MIDDLE_POS_DEG: dict[str, tuple[float, ...]] = {
    "right": (25, -20, 25, -15, 20, -25, 15, -15),
    "left": (20, -10, 33, -35, 30, -10, 20, -8),
}


def hand_pose_deg(side: str, closed: bool) -> tuple[float, ...]:
    """Return the 8 per-servo target angles (deg) for a fully open or closed hand.

    Values are relative joint targets only; they do NOT include the `MIDDLE_POS_DEG`
    per-servo zero trim, which is a calibration offset applied at the tick layer.
    """
    if side not in HAND_SIDES:
        raise ValueError(f"Unknown hand side: {side!r}")
    pose: list[float] = []
    for finger in range(1, 5):
        if closed and finger == THUMB_FINGER_INDEX:
            pose.extend(THUMB_CLOSE_DEG)
        elif closed:
            pose.extend(HAND_CLOSE_DEG)
        else:
            pose.extend(HAND_OPEN_DEG)
    return tuple(pose)


def hand_pose_rad(side: str, closed: bool) -> tuple[float, ...]:
    """Return `hand_pose_deg` converted to radians."""
    return tuple(math.radians(deg) for deg in hand_pose_deg(side, closed))


DEFAULT_HEAD_Q: tuple[float, float] = (0.0, 0.0)
DEFAULT_HAND_Q: dict[str, tuple[float, ...]] = {
    side: hand_pose_rad(side, closed=False) for side in HAND_SIDES
}


def default_action() -> dict[str, float]:
    """Return a default action dict with all 41 keys: zero body/head, hands open."""
    action = dict.fromkeys(BODY_KEYS, 0.0)
    action.update(zip(HEAD_KEYS, DEFAULT_HEAD_Q, strict=True))
    action.update(zip(LEFT_HAND_KEYS, DEFAULT_HAND_Q["left"], strict=True))
    action.update(zip(RIGHT_HAND_KEYS, DEFAULT_HAND_Q["right"], strict=True))
    return action
