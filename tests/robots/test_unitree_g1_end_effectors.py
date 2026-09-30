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

"""Tests for the Unitree G1 joint/motor layout tables: body slots, heads and end effectors. No hardware required."""

import math

import pytest

from lerobot.robots.unitree_g1 import end_effectors as ee, g1_utils as g1, heads
from lerobot.robots.unitree_g1.headhand import LEGACY_MOTOR_NAMES
from lerobot.robots.unitree_g1.headhand_devices import HEAD_HAND_MOTORS
from lerobot.teleoperators.unitree_g1_gamepad.unitree_g1_gamepad import TELEOP_ACTION_KEYS, default_targets

AH = ee.AMAZING_HAND


def test_key_counts():
    assert len(g1.BODY_KEYS) == 29
    assert len(g1.ARM_KEYS) == 14
    assert len(heads.HEAD_KEYS) == 2
    assert len(AH.joint_keys("left")) == 8
    assert len(AH.joint_keys("right")) == 8
    assert len(AH.joint_keys()) == 16
    assert len(ee.DEX3.joint_keys()) == 14
    assert len(ee.DEX1.joint_keys()) == 2
    assert len(ee.ALL_HAND_KEYS) == 32
    assert len(TELEOP_ACTION_KEYS) == 89
    assert ee.HAND_CLOSURE_KEYS == ("kLeftHand.closure", "kRightHand.closure")


def test_body_and_arm_keys_match_joint_indices():
    assert tuple(f"{joint.name}.q" for joint in g1.G1_29_JointIndex) == g1.BODY_KEYS
    assert tuple(f"{joint.name}.q" for joint in g1.G1_29_JointArmIndex) == g1.ARM_KEYS


def test_teleop_action_keys_layout():
    expected = (
        g1.BODY_KEYS
        + heads.HEAD_KEYS
        + ee.ALL_HAND_KEYS
        + ee.HAND_CLOSURE_KEYS
        + g1.REMOTE_KEYS
        + g1.NAV_KEYS
        + (g1.BASE_HEIGHT_KEY,)
    )
    assert expected == TELEOP_ACTION_KEYS
    assert len(set(TELEOP_ACTION_KEYS)) == len(TELEOP_ACTION_KEYS)


def test_default_targets_zero_body_centered_head_open_hands():
    targets = default_targets()
    assert set(targets) == set(g1.BODY_KEYS + heads.HEAD_KEYS + ee.ALL_HAND_KEYS)
    assert all(targets[key] == 0.0 for key in g1.BODY_KEYS)
    assert [targets[key] for key in heads.HEAD_KEYS] == list(heads.DEFAULT_HEAD_Q)
    for spec in ee.HAND_SPECS.values():
        for side in ee.HAND_SIDES:
            assert tuple(targets[key] for key in spec.joint_keys(side)) == spec.open_q[side]


def test_head_and_hand_names_follow_unitree_convention():
    assert heads.HEAD_KEYS == ("kHeadYaw.q", "kHeadPitch.q")
    assert ee.amazing_hand_motor_names("left") == tuple(f"kLeftHandMotor{i}" for i in range(11, 19))
    assert ee.amazing_hand_motor_names("right") == tuple(f"kRightHandMotor{i}" for i in range(1, 9))
    for name, (motor_id, _model) in ee.AMAZING_HAND_MOTORS.items():
        assert name.endswith(f"Motor{motor_id}")
    assert ee.DEX3.motor_names["left"][0] == "kLeftHandThumb0"
    assert ee.DEX1.joint_keys() == ("kLeftGripper.q", "kRightGripper.q")


def test_legacy_motor_names_map_onto_every_current_motor():
    assert set(LEGACY_MOTOR_NAMES.values()) == set(HEAD_HAND_MOTORS)
    assert LEGACY_MOTOR_NAMES["xl330_joint"] == "kHeadYaw"
    assert LEGACY_MOTOR_NAMES["d455_joint"] == "kHeadPitch"
    assert LEGACY_MOTOR_NAMES["left_hand_finger1_motor1"] == "kLeftHandMotor11"
    assert LEGACY_MOTOR_NAMES["right_hand_finger4_motor2"] == "kRightHandMotor8"


def test_hand_closure_key_rejects_unknown_side():
    with pytest.raises(ValueError):
        ee.hand_closure_key("middle")
    with pytest.raises(ValueError):
        ee.amazing_hand_motor_names("middle")


def test_arm_keys_subset_of_body_keys():
    assert set(g1.ARM_KEYS) <= set(g1.BODY_KEYS)


def test_invalid_body_keys_are_the_six_missing_slots_of_body_keys():
    keys = g1.invalid_body_keys("23dof")
    assert set(keys) <= set(g1.BODY_KEYS)
    assert len(keys) == 6
    assert [g1.BODY_KEYS.index(key) for key in keys] == list(g1.G1_23_INVALID_SDK_SLOTS)
    assert g1.invalid_sdk_slots("23dof") == g1.G1_23_INVALID_SDK_SLOTS


def test_29dof_body_has_no_invalid_slots():
    assert g1.invalid_body_keys("29dof") == ()
    assert g1.invalid_sdk_slots("29dof") == ()


def test_leg_slots_are_the_first_twelve():
    assert tuple(range(12)) == g1.G1_LEG_SLOTS
    assert all(
        "Hip" in g1.BODY_KEYS[i] or "Knee" in g1.BODY_KEYS[i] or "Ankle" in g1.BODY_KEYS[i]
        for i in g1.G1_LEG_SLOTS
    )


def test_hand_names_and_ids_unique():
    left_names = ee.amazing_hand_motor_names("left")
    right_names = ee.amazing_hand_motor_names("right")
    assert len(set(left_names) | set(right_names)) == 16
    assert ee.AMAZING_HAND_MOTOR_IDS["right"] == tuple(range(1, 9))
    assert ee.AMAZING_HAND_MOTOR_IDS["left"] == tuple(range(11, 19))


def test_head_hand_motors_unique_names_and_ids():
    assert len(HEAD_HAND_MOTORS) == 18
    ids = [(name, motor[0]) for name, motor in HEAD_HAND_MOTORS.items()]
    assert len(ids) == len(set(ids))
    assert set(HEAD_HAND_MOTORS) == set(heads.HEAD_MOTORS) | set(ee.AMAZING_HAND_MOTORS)


def test_keys_have_no_slash_and_end_with_q_except_remote_closure_and_cmd():
    for key in TELEOP_ACTION_KEYS:
        assert "/" not in key
        if key in ee.HAND_CLOSURE_KEYS:
            assert key.endswith(".closure")
        elif key == g1.BASE_HEIGHT_KEY or key in g1.NAV_KEYS:
            assert key.endswith(".cmd")
        elif key not in g1.REMOTE_KEYS:
            assert key.endswith(".q")


def test_all_hand_keys_have_no_duplicates():
    assert len(set(ee.ALL_HAND_KEYS)) == len(ee.ALL_HAND_KEYS)
    assert not set(ee.ALL_HAND_KEYS) & set(g1.BODY_KEYS + heads.HEAD_KEYS)


def test_hand_specs_registry():
    assert set(ee.HAND_SPECS) == set(ee.END_EFFECTORS) - {"none"}
    for name, spec in ee.HAND_SPECS.items():
        assert spec.name == name
        for side in ee.HAND_SIDES:
            assert len(spec.open_q[side]) == len(spec.closed_q[side]) == len(spec.motor_names[side])


def test_amazing_hand_pose_deg_open():
    assert ee.amazing_hand_pose_deg(False) == ee.AMAZING_HAND_OPEN_DEG * 4


def test_amazing_hand_pose_deg_closed():
    pose = ee.amazing_hand_pose_deg(True)
    assert len(pose) == 8
    thumb_start = (ee.AMAZING_HAND_THUMB_FINGER - 1) * 2
    assert pose[thumb_start : thumb_start + 2] == ee.AMAZING_HAND_THUMB_CLOSE_DEG
    non_thumb = pose[:thumb_start] + pose[thumb_start + 2 :]
    assert non_thumb == ee.AMAZING_HAND_CLOSE_DEG * 3


@pytest.mark.parametrize("side", ["left", "right"])
def test_amazing_hand_spec_poses_are_the_deg_poses(side):
    assert AH.open_q[side] == pytest.approx([math.radians(d) for d in ee.amazing_hand_pose_deg(False)])
    assert AH.closed_q[side] == pytest.approx([math.radians(d) for d in ee.amazing_hand_pose_deg(True)])


def test_amazing_hand_poses_within_limit():
    for side in ee.HAND_SIDES:
        for pose in (AH.open_q[side], AH.closed_q[side]):
            for value in pose:
                assert abs(value) <= ee.AMAZING_HAND_LIMIT_RAD + 1e-9


def test_amazing_hand_middle_pos_has_one_trim_per_servo():
    for side in ee.HAND_SIDES:
        assert len(ee.AMAZING_HAND_MIDDLE_POS_DEG[side]) == len(ee.AMAZING_HAND_MOTOR_IDS[side])


@pytest.mark.parametrize("side", ["left", "right"])
def test_closure_endpoints_match_open_and_closed_poses(side):
    assert AH.closure_to_q(side, 0.0) == pytest.approx(AH.open_q[side])
    assert AH.closure_to_q(side, 1.0) == pytest.approx(AH.closed_q[side])


@pytest.mark.parametrize("side", ["left", "right"])
def test_closure_midpoint_is_per_motor_mean(side):
    mid = AH.closure_to_q(side, 0.5)
    for q, open_q, closed_q in zip(mid, AH.open_q[side], AH.closed_q[side], strict=True):
        assert q == pytest.approx((open_q + closed_q) / 2)


@pytest.mark.parametrize("closure, expected", [(-0.5, 0.0), (1.5, 1.0)])
def test_closure_to_q_clips(closure, expected):
    assert AH.closure_to_q("right", closure) == pytest.approx(AH.closure_to_q("right", expected))


@pytest.mark.parametrize("spec", [ee.AMAZING_HAND, ee.DEX3, ee.DEX1], ids=lambda spec: spec.name)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("closure", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_q_to_closure_inverts_closure_to_q(spec, side, closure):
    assert spec.q_to_closure(side, spec.closure_to_q(side, closure)) == pytest.approx(closure)


def test_q_to_closure_clips_out_of_range_angles():
    beyond_open = [q + (q - c) for q, c in zip(AH.open_q["right"], AH.closed_q["right"], strict=True)]
    beyond_closed = [c + (c - q) for q, c in zip(AH.open_q["right"], AH.closed_q["right"], strict=True)]
    assert AH.q_to_closure("right", beyond_open) == 0.0
    assert AH.q_to_closure("right", beyond_closed) == 1.0


def test_q_to_closure_averages_uneven_fingers():
    q = list(AH.open_q["left"])
    q[:2] = AH.closed_q["left"][:2]
    assert AH.q_to_closure("left", q) == pytest.approx(2 / 8)


@pytest.mark.parametrize("spec", [ee.AMAZING_HAND, ee.DEX3, ee.DEX1], ids=lambda spec: spec.name)
def test_q_to_closure_rejects_wrong_length(spec):
    with pytest.raises(ValueError):
        spec.q_to_closure("left", [0.0] * (len(spec.open_q["left"]) + 1))


def test_dex3_joint_keys():
    assert ee.DEX3.joint_keys("left") == tuple(f"kLeftHand{joint}.q" for joint in ee.DEX3_JOINTS)
    assert ee.DEX3.joint_keys("right") == tuple(f"kRightHand{joint}.q" for joint in ee.DEX3_JOINTS)
    assert ee.DEX3.joint_keys() == ee.DEX3.joint_keys("left") + ee.DEX3.joint_keys("right")


def test_dex3_right_hand_flexes_the_other_way():
    assert ee.DEX3.open_q["left"] == ee.DEX3.open_q["right"]
    assert ee.DEX3.closed_q["right"] == tuple(-q for q in ee.DEX3.closed_q["left"])
    # Flexing joints close positive on one side and negative on the other.
    for left, right in zip(ee.DEX3.closed_q["left"], ee.DEX3.closed_q["right"], strict=True):
        assert left == 0.0 or math.copysign(1, left) != math.copysign(1, right)


def test_dex3_closure_ignores_the_non_flexing_joint():
    # Thumb0 is 0.0 in both poses, so its value must not affect the closure.
    q = list(ee.DEX3.closure_to_q("right", 0.5))
    q[0] = 1.0
    assert ee.DEX3.q_to_closure("right", q) == pytest.approx(0.5)


def test_dex1_closure_maps_onto_finger_travel():
    assert ee.DEX1.closure_to_q("left", 0.0) == (0.0245,)
    assert ee.DEX1.closure_to_q("right", 1.0) == (pytest.approx(-0.02),)
    assert ee.DEX1.q_to_closure("left", (0.0245,)) == 0.0
    assert ee.DEX1.q_to_closure("left", (-0.03,)) == 1.0  # past contact clips


def test_mode_machine_by_revision():
    assert set(g1.MODE_MACHINE_BY_REVISION) == set(g1.REVISIONS)
    assert g1.MODE_MACHINE_BY_REVISION["base"] == 1
    assert g1.MODE_MACHINE_BY_REVISION["rev_1_0"] == 4
    assert g1.BODIES == ("29dof", "23dof")


def test_head_limits_and_default():
    assert set(heads.HEAD_LIMITS_RAD) == set(heads.HEAD_MOTORS)
    for name, q in zip(heads.HEAD_MOTORS, heads.DEFAULT_HEAD_Q, strict=True):
        low, high = heads.HEAD_LIMITS_RAD[name]
        assert low <= q <= high
    assert heads.HEADS == ("none", "d455_pan_tilt")
