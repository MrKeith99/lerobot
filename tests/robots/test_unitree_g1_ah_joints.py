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

"""Tests for the UnitreeG1Ah joint/motor layout tables. No hardware required."""

import pytest

from lerobot.robots.unitree_g1_ah import g1_ah_joints as j


def test_key_counts():
    assert len(j.BODY_KEYS) == 23
    assert len(j.ARM_KEYS) == 10
    assert len(j.HEAD_KEYS) == 2
    assert len(j.LEFT_HAND_KEYS) == 8
    assert len(j.RIGHT_HAND_KEYS) == 8
    assert len(j.HAND_KEYS) == 16
    assert len(j.ALL_ACTION_KEYS) == 41
    assert len(j.ARM_MODE_ACTION_KEYS) == 32
    assert len(j.TELEOP_ACTION_KEYS) == 63
    assert len(j.HEAD_HAND_KEYS) == 18
    assert j.HAND_CLOSURE_KEYS == ("left_hand.closure", "right_hand.closure")
    assert len(j.CLOSURE_ACTION_KEYS) == 27
    assert len(j.CLOSURE_ARM_MODE_ACTION_KEYS) == 18


def test_closure_key_order_body_head_left_right():
    assert j.CLOSURE_ACTION_KEYS == j.BODY_KEYS + j.HEAD_KEYS + ("left_hand.closure", "right_hand.closure")
    assert j.CLOSURE_ARM_MODE_ACTION_KEYS[-4:] == j.REMOTE_AXES
    assert set(j.CLOSURE_ACTION_KEYS) <= set(j.TELEOP_ACTION_KEYS)
    assert set(j.CLOSURE_ARM_MODE_ACTION_KEYS) <= set(j.TELEOP_ACTION_KEYS)


def test_hand_closure_key_rejects_unknown_side():
    with pytest.raises(ValueError):
        j.hand_closure_key("middle")


def test_sdk23_to_29_strictly_increasing_and_valid():
    assert list(j.SDK23_TO_29) == sorted(j.SDK23_TO_29)
    assert len(set(j.SDK23_TO_29)) == len(j.SDK23_TO_29)
    for slot in j.SDK23_TO_29:
        assert slot not in j.G1_23_INVALID_SDK_SLOTS
    assert len(j.SDK23_TO_29) == 23


def test_sdk29_to_23_is_inverse():
    for sdk23, sdk29 in enumerate(j.SDK23_TO_29):
        assert j.SDK29_TO_23[sdk29] == sdk23
    assert len(j.SDK29_TO_23) == 23


def test_body23_sdk29_round_trip():
    x = [float(i) for i in range(23)]
    assert j.body23_from_sdk29(j.sdk29_from_body23(x)) == x


def test_sdk29_from_body23_fill_at_invalid_slots():
    x = [1.0] * 23
    out = j.sdk29_from_body23(x, fill=-1.0)
    assert len(out) == 29
    for slot in j.G1_23_INVALID_SDK_SLOTS:
        assert out[slot] == -1.0
    for slot in j.SDK23_TO_29:
        assert out[slot] == 1.0


def test_length_validation_raises():
    with pytest.raises(ValueError):
        j.body23_from_sdk29([0.0] * 28)
    with pytest.raises(ValueError):
        j.body23_from_sdk29([0.0] * 30)
    with pytest.raises(ValueError):
        j.sdk29_from_body23([0.0] * 22)
    with pytest.raises(ValueError):
        j.sdk29_from_body23([0.0] * 24)


def test_key_order_body_head_left_right():
    expected = j.BODY_KEYS + j.HEAD_KEYS + j.LEFT_HAND_KEYS + j.RIGHT_HAND_KEYS
    assert expected == j.ALL_ACTION_KEYS


def test_arm_keys_subset_of_body_keys():
    assert set(j.ARM_KEYS) <= set(j.BODY_KEYS)


def test_invalid_body_keys_disjoint_from_body_keys():
    assert set(j.INVALID_BODY_KEYS) & set(j.BODY_KEYS) == set()
    assert len(j.INVALID_BODY_KEYS) == 6


def test_hand_names_and_ids_unique():
    left_names = j.hand_motor_names("left")
    right_names = j.hand_motor_names("right")
    assert len(set(left_names) | set(right_names)) == 16
    assert j.HAND_MOTOR_IDS["right"] == tuple(range(1, 9))
    assert j.HAND_MOTOR_IDS["left"] == tuple(range(11, 19))


def test_head_hand_motors_unique_names_and_ids():
    assert len(j.HEAD_HAND_MOTORS) == 18
    ids = [(name, motor[0]) for name, motor in j.HEAD_HAND_MOTORS.items()]
    assert len(ids) == len(set(ids))
    assert len(j.HEAD_HAND_MOTORS) == len(set(j.HEAD_HAND_MOTORS.keys()))


def test_keys_have_no_slash_and_end_with_q_except_remote_and_closure():
    for key in j.TELEOP_ACTION_KEYS:
        assert "/" not in key
        if key in j.HAND_CLOSURE_KEYS:
            assert key.endswith(".closure")
        elif key not in j.REMOTE_KEYS:
            assert key.endswith(".q")


def test_slices_for_all_action_keys():
    slices = j.slices_for(j.ALL_ACTION_KEYS)
    assert slices.body == slice(0, 23)
    assert slices.head == slice(23, 25)
    assert slices.left_hand == slice(25, 33)
    assert slices.right_hand == slice(33, 41)


def test_slices_for_raises_on_non_contiguous():
    shuffled = (j.BODY_KEYS[0],) + j.HEAD_KEYS + (j.BODY_KEYS[1],) + j.LEFT_HAND_KEYS + j.RIGHT_HAND_KEYS
    with pytest.raises(ValueError):
        j.slices_for(shuffled)


def test_action_vector_round_trip():
    action = j.default_action()
    vec = j.action_to_vector(action, j.ALL_ACTION_KEYS)
    assert len(vec) == 41
    round_tripped = j.vector_to_action(vec, j.ALL_ACTION_KEYS)
    assert round_tripped == action


def test_action_to_vector_missing_key_error_lists_missing():
    action = j.default_action()
    del action[j.BODY_KEYS[0]]
    del action[j.HEAD_KEYS[0]]
    with pytest.raises(KeyError) as exc_info:
        j.action_to_vector(action, j.ALL_ACTION_KEYS)
    message = str(exc_info.value)
    assert j.BODY_KEYS[0] in message
    assert j.HEAD_KEYS[0] in message


def test_hand_pose_deg_open():
    assert j.hand_pose_deg("right", False) == j.HAND_OPEN_DEG * 4
    assert j.hand_pose_deg("left", False) == j.HAND_OPEN_DEG * 4


def test_hand_pose_deg_closed():
    pose = j.hand_pose_deg("right", True)
    assert len(pose) == 8
    thumb_start = (j.THUMB_FINGER_INDEX - 1) * 2
    thumb_pose = pose[thumb_start : thumb_start + 2]
    assert thumb_pose == j.THUMB_CLOSE_DEG
    non_thumb = pose[:thumb_start] + pose[thumb_start + 2 :]
    assert non_thumb == j.HAND_CLOSE_DEG * 3


def test_hand_pose_rad_within_limit():
    for side in j.HAND_SIDES:
        for closed in (False, True):
            for value in j.hand_pose_rad(side, closed):
                assert abs(value) <= j.HAND_LIMIT_RAD + 1e-9


@pytest.mark.parametrize("side", ["left", "right"])
def test_closure_endpoints_match_open_and_closed_poses(side):
    assert j.closure_to_hand_q(side, 0.0) == pytest.approx(j.hand_pose_rad(side, False))
    assert j.closure_to_hand_q(side, 1.0) == pytest.approx(j.hand_pose_rad(side, True))


@pytest.mark.parametrize("side", ["left", "right"])
def test_closure_midpoint_is_per_motor_mean(side):
    mid = j.closure_to_hand_q(side, 0.5)
    for q, open_q, closed_q in zip(
        mid, j.hand_pose_rad(side, False), j.hand_pose_rad(side, True), strict=True
    ):
        assert q == pytest.approx((open_q + closed_q) / 2)


@pytest.mark.parametrize("closure, expected", [(-0.5, 0.0), (1.5, 1.0)])
def test_closure_to_hand_q_clips(closure, expected):
    assert j.closure_to_hand_q("right", closure) == pytest.approx(j.closure_to_hand_q("right", expected))


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("closure", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_hand_q_to_closure_inverts_closure_to_hand_q(side, closure):
    assert j.hand_q_to_closure(side, j.closure_to_hand_q(side, closure)) == pytest.approx(closure)


def test_hand_q_to_closure_clips_out_of_range_angles():
    beyond_open = [
        q + (q - c)
        for q, c in zip(j.hand_pose_rad("right", False), j.hand_pose_rad("right", True), strict=True)
    ]
    beyond_closed = [
        c + (c - q)
        for q, c in zip(j.hand_pose_rad("right", False), j.hand_pose_rad("right", True), strict=True)
    ]
    assert j.hand_q_to_closure("right", beyond_open) == 0.0
    assert j.hand_q_to_closure("right", beyond_closed) == 1.0


def test_hand_q_to_closure_averages_uneven_fingers():
    q = list(j.hand_pose_rad("left", False))
    q[:2] = j.hand_pose_rad("left", True)[:2]
    assert j.hand_q_to_closure("left", q) == pytest.approx(2 / 8)


def test_hand_q_to_closure_rejects_wrong_length():
    with pytest.raises(ValueError):
        j.hand_q_to_closure("left", [0.0] * 7)


def test_default_action_keys_and_zero_body():
    action = j.default_action()
    assert set(action.keys()) == set(j.ALL_ACTION_KEYS)
    for key in j.BODY_KEYS:
        assert action[key] == 0.0


def test_robot_type_and_mode_machine_by_revision():
    assert j.ROBOT_TYPE_BY_REVISION["rev_1_0"] == "unitree_g1_23dof_ah8_d455_2dof_rev_1_0"
    assert j.ROBOT_TYPE_BY_REVISION["base"] == "unitree_g1_23dof_ah8_d455_2dof"
    assert j.MODE_MACHINE_BY_REVISION["base"] == 1
    assert j.MODE_MACHINE_BY_REVISION["rev_1_0"] == 4
