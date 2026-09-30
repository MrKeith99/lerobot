#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import importlib
from enum import IntEnum

import numpy as np

# ruff: noqa: N801, N815

NUM_MOTORS = 29

REMOTE_AXES = ("remote.lx", "remote.ly", "remote.rx", "remote.ry")
REMOTE_BUTTONS = tuple(f"remote.button.{i}" for i in range(16))
REMOTE_KEYS = REMOTE_AXES + REMOTE_BUTTONS


# Absolute GR00T base-height command in m (the policy's height observation), recorded as an action
# when `UnitreeG1Config.base_height_action` is on. Remote R1/R2 move it at GROOT_BASE_HEIGHT_RATE.
BASE_HEIGHT_KEY = "kBaseHeight.cmd"
GROOT_BASE_HEIGHT_DEFAULT = 0.74
GROOT_BASE_HEIGHT_RANGE = (0.50, 1.00)
GROOT_BASE_HEIGHT_RATE = 0.05  # m/s


# Absolute navigation command (vx, vy, yaw rate) consumed by the locomotion controllers, recorded as
# an action in controller mode. Derived from the remote sticks as (ly, -lx, -rx).
NAV_KEYS = ("kNavVx.cmd", "kNavVy.cmd", "kNavYawRate.cmd")
# Feedback entry the record/teleoperate scripts add to a G1 observation for the teleoperator: the robot's
# `robot_type`, e.g. for an XR teleop to refuse a robot of another embodiment.
ROBOT_TYPE_FEEDBACK_KEY = "robot_type"


def nav_from_remote(lx: float, ly: float, rx: float) -> dict[str, float]:
    """Map remote stick axes to the navigation command, the mapping both controllers use."""
    return dict(zip(NAV_KEYS, (float(ly), -float(lx), -float(rx)), strict=True))


def nav_command(action: dict) -> tuple[float, float, float]:
    """(vx, vy, yaw rate) from explicit NAV_KEYS if all are present, else from the remote sticks."""
    if all(key in action for key in NAV_KEYS):
        return tuple(float(action[key]) for key in NAV_KEYS)
    lx, ly, rx, _ry = (action.get(k, 0.0) for k in REMOTE_AXES)
    return tuple(nav_from_remote(lx, ly, rx).values())


def default_remote_input() -> dict[str, float]:
    """Return a zeroed-out remote input dict (axes + buttons)."""
    return dict.fromkeys(REMOTE_KEYS, 0.0)


def get_gravity_orientation(quaternion: list[float] | np.ndarray) -> np.ndarray:
    """Get gravity orientation from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


def make_locomotion_controller(name: str | None):
    """Instantiate a locomotion controller by class name. Returns None if name is None."""
    if name is None:
        return None
    controllers = {
        "GrootLocomotionController": "lerobot.robots.unitree_g1.gr00t_locomotion",
        "HolosomaLocomotionController": "lerobot.robots.unitree_g1.holosoma_locomotion",
    }
    module_path = controllers.get(name)
    if module_path is None:
        raise ValueError(f"Unknown controller: {name!r}. Available: {list(controllers)}")
    module = importlib.import_module(module_path)
    return getattr(module, name)()


class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


# Body variants. Both keep the 29-slot SDK layout; the 23dof body (rev_1_0 hardware) has no
# waist roll/pitch and no wrist pitch/yaw, so those slots are recorded as 0.0 and never commanded.
BODIES = ("29dof", "23dof")
REVISIONS = ("base", "rev_1_0")
MODE_MACHINE_BY_REVISION = {"base": 1, "rev_1_0": 4}
G1_23_INVALID_SDK_SLOTS = (13, 14, 20, 21, 27, 28)
G1_LEG_SLOTS = tuple(range(12))

BODY_KEYS = tuple(f"{joint.name}.q" for joint in G1_29_JointIndex)
ARM_KEYS = tuple(f"{joint.name}.q" for joint in G1_29_JointArmIndex)


def invalid_sdk_slots(body: str) -> tuple[int, ...]:
    """SDK slots the given body has no motor for."""
    return G1_23_INVALID_SDK_SLOTS if body == "23dof" else ()


def invalid_body_keys(body: str) -> tuple[str, ...]:
    """`.q` keys of the SDK slots the given body has no motor for."""
    return tuple(f"{G1_29_JointIndex(slot).name}.q" for slot in invalid_sdk_slots(body))
