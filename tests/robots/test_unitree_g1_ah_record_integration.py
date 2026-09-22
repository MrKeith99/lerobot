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

"""End-to-end record-loop contract test for UnitreeG1Ah: get_observation -> teleop get_action ->
send_action -> build_dataset_frame, with no hardware and no real Unitree SDK.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from lerobot.utils.import_utils import _unitree_sdk_available

if not _unitree_sdk_available:
    pytest.skip("Unitree SDK not available", allow_module_level=True)

from lerobot.robots.unitree_g1_ah.g1_ah_joints import ALL_ACTION_KEYS, ARM_MODE_ACTION_KEYS, HEAD_HAND_MOTORS
from lerobot.teleoperators.unitree_g1_ah_gamepad import (
    UnitreeG1AhGamepadTeleop,
    UnitreeG1AhGamepadTeleopConfig,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from tests.mocks.mock_unitree_g1_ah_server import MockHeadHandServer
from tests.robots.test_unitree_g1_ah import _make_g1ah, _make_sdk_mocks, _make_stub_controller, _new_robot
from tests.teleoperators.test_unitree_g1_ah_gamepad import FakeInput

_GAMEPAD_MODULE = "lerobot.teleoperators.unitree_g1_ah_gamepad.unitree_g1_ah_gamepad"


@pytest.fixture
def headhand_server():
    with MockHeadHandServer() as server:
        yield server


@pytest.fixture
def teleop():
    with patch(f"{_GAMEPAD_MODULE}.UnitreeG1AhGamepadInput", FakeInput):
        t = UnitreeG1AhGamepadTeleop(UnitreeG1AhGamepadTeleopConfig())
        t.connect()
        yield t
        if t.is_connected:
            t.disconnect()


class TestG1AhRecordLoopIntegration:
    def test_full_dof_record_loop(self, headhand_server, teleop, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(headhand_server, mocks, tmp_path)
            robot.connect(calibrate=False)

            ds_features = combine_feature_dicts(
                hw_to_dataset_features(robot.action_features, ACTION),
                hw_to_dataset_features(robot.observation_features, OBS_STR),
            )
            assert ds_features[ACTION]["names"] == list(ALL_ACTION_KEYS)
            assert ds_features["observation.state"]["names"] == list(ALL_ACTION_KEYS)

            try:
                for _ in range(5):
                    obs = robot.get_observation()
                    act = teleop.get_action()
                    sent = robot.send_action(act)

                    obs_frame = build_dataset_frame(ds_features, obs, OBS_STR)
                    act_frame = build_dataset_frame(ds_features, act, ACTION)

                    assert obs_frame["observation.state"].shape == (41,)
                    assert act_frame[ACTION].shape == (41,)

                    for name in HEAD_HAND_MOTORS:
                        key = f"{name}.q"
                        idx = ds_features[ACTION]["names"].index(key)
                        assert act_frame[ACTION][idx] == pytest.approx(sent[key], abs=1e-3)
            finally:
                robot.disconnect()

    def test_controller_mode_record_loop(self, headhand_server, teleop, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        controller = _make_stub_controller()
        patches, *_ = _make_g1ah(mocks, controller=controller)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(
                headhand_server, mocks, tmp_path, config_kwargs={"controller": "GrootLocomotionController"}
            )
            robot.connect(calibrate=False)

            ds_features = combine_feature_dicts(
                hw_to_dataset_features(robot.action_features, ACTION),
                hw_to_dataset_features(robot.observation_features, OBS_STR),
            )
            assert ds_features[ACTION]["names"] == list(ARM_MODE_ACTION_KEYS)

            try:
                for _ in range(5):
                    obs = robot.get_observation()
                    action = teleop.get_action()
                    sent = robot.send_action(action)

                    act_frame = build_dataset_frame(ds_features, sent, ACTION)
                    assert act_frame[ACTION].shape == (32,)

                    obs_frame = build_dataset_frame(ds_features, obs, OBS_STR)
                    assert obs_frame["observation.state"].shape == (41,)
            finally:
                robot.disconnect()
