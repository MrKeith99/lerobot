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

"""End-to-end record-loop contract test for the Unitree G1 AmazingHand embodiment: get_observation -> teleop get_action ->
send_action -> build_dataset_frame, with no hardware and no real Unitree SDK.
"""

from __future__ import annotations

import contextlib
import time
from unittest.mock import patch

import pytest

from lerobot.utils.import_utils import _unitree_sdk_available

if not _unitree_sdk_available:
    pytest.skip("Unitree SDK not available", allow_module_level=True)

from lerobot.robots.unitree_g1.end_effectors import HAND_CLOSURE_KEYS
from lerobot.robots.unitree_g1.g1_utils import ARM_KEYS, BASE_HEIGHT_KEY, BODY_KEYS, NAV_KEYS
from lerobot.robots.unitree_g1.headhand_devices import HEAD_HAND_MOTORS
from lerobot.teleoperators.unitree_g1_gamepad import (
    UnitreeG1GamepadTeleop,
    UnitreeG1GamepadTeleopConfig,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from tests.mocks.mock_unitree_g1_headhand_server import MockHeadHandServer
from tests.robots.test_unitree_g1_amazing_hand import (
    _make_patches,
    _make_sdk_mocks,
    _make_stub_controller,
    _new_robot,
    _state_keys,
)
from tests.teleoperators.test_unitree_g1_gamepad import FakeInput

_GAMEPAD_MODULE = "lerobot.teleoperators.unitree_g1_gamepad.unitree_g1_gamepad"


@pytest.fixture
def headhand_server():
    with MockHeadHandServer() as server:
        yield server


@pytest.fixture
def teleop():
    with patch(f"{_GAMEPAD_MODULE}.UnitreeG1GamepadInput", FakeInput):
        t = UnitreeG1GamepadTeleop(UnitreeG1GamepadTeleopConfig())
        t.connect()
        yield t
        if t.is_connected:
            t.disconnect()


def _dataset_features(robot):
    return combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION),
        hw_to_dataset_features(robot.observation_features, OBS_STR),
    )


class TestAmazingHandRecordLoopIntegration:
    @pytest.mark.parametrize("hand_representation", ["closure", "per_motor"])
    def test_full_dof_record_loop(self, headhand_server, teleop, tmp_path, hand_representation):
        state_keys = _state_keys(hand_representation)
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_patches(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(
                headhand_server, mocks, tmp_path, config_kwargs={"hand_representation": hand_representation}
            )
            robot.connect(calibrate=False)
            try:
                ds_features = _dataset_features(robot)
                assert ds_features[ACTION]["names"] == state_keys
                assert ds_features["observation.state"]["names"] == state_keys

                for _ in range(5):
                    obs = robot.get_observation()
                    act = teleop.get_action()
                    sent = robot.send_action(act)

                    obs_frame = build_dataset_frame(ds_features, obs, OBS_STR)
                    act_frame = build_dataset_frame(ds_features, act, ACTION)

                    assert obs_frame["observation.state"].shape == (len(state_keys),)
                    assert act_frame[ACTION].shape == (len(state_keys),)

                    recorded = HAND_CLOSURE_KEYS if hand_representation == "closure" else ()
                    recorded += tuple(f"{name}.q" for name in HEAD_HAND_MOTORS if f"{name}.q" in state_keys)
                    for key in recorded:
                        idx = ds_features[ACTION]["names"].index(key)
                        assert act_frame[ACTION][idx] == pytest.approx(sent[key], abs=1e-3)
            finally:
                robot.disconnect()

    @pytest.mark.parametrize("hand_representation", ["closure", "per_motor"])
    def test_controller_mode_record_loop(self, headhand_server, teleop, tmp_path, hand_representation):
        state_keys = _state_keys(hand_representation, controller=True)
        action_keys = [*state_keys, *NAV_KEYS, BASE_HEIGHT_KEY]
        mocks = _make_sdk_mocks(mode_machine=4)
        controller = _make_stub_controller()
        patches, *_ = _make_patches(mocks, controller=controller)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(
                headhand_server,
                mocks,
                tmp_path,
                config_kwargs={
                    "controller": "GrootLocomotionController",
                    "hand_representation": hand_representation,
                },
            )
            robot.connect(calibrate=False)
            try:
                ds_features = _dataset_features(robot)
                assert ds_features[ACTION]["names"] == action_keys
                assert ds_features["observation.state"]["names"] == state_keys

                for _ in range(5):
                    obs = robot.get_observation()
                    action = teleop.get_action()
                    sent = robot.send_action(action)

                    act_frame = build_dataset_frame(ds_features, sent, ACTION)
                    assert act_frame[ACTION].shape == (len(action_keys),)
                    assert act_frame[ACTION][-1] == pytest.approx(0.74)

                    obs_frame = build_dataset_frame(ds_features, obs, OBS_STR)
                    assert obs_frame["observation.state"].shape == (len(state_keys),)
                    assert all(key in obs for key in BODY_KEYS)
                    teleop_frame = build_dataset_frame(ds_features, action, ACTION)[ACTION]
                    for key in NAV_KEYS:
                        assert key in ds_features[ACTION]["names"]
                        assert teleop_frame[ds_features[ACTION]["names"].index(key)] == pytest.approx(0.0)
            finally:
                robot.disconnect()

    def test_held_rb_is_recorded_as_right_closure(self, headhand_server, teleop, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_patches(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(headhand_server, mocks, tmp_path)
            robot.connect(calibrate=False)
            try:
                ds_features = _dataset_features(robot)
                names = ds_features[ACTION]["names"]
                clock = {"t": 0.0}
                with patch(f"{_GAMEPAD_MODULE}.time.perf_counter", lambda: clock["t"]):
                    teleop.get_action()
                    teleop.gamepad.buttons.add(teleop.config.layout.button_rb)
                    closures = []
                    for _ in range(30):
                        clock["t"] += 0.05
                        act = teleop.get_action()
                        robot.send_action(act)
                        frame = build_dataset_frame(ds_features, act, ACTION)[ACTION]
                        closures.append(float(frame[names.index("kRightHand.closure")]))
                        assert frame[names.index("kLeftHand.closure")] == 0.0

                assert closures == sorted(closures)
                assert closures[0] == pytest.approx(teleop.config.hand_blend_per_s * 0.05)
                assert closures[-1] == pytest.approx(1.0)

                deadline = time.time() + 2.0
                obs = robot.get_observation()
                while time.time() < deadline and obs.get("kRightHand.closure", 0.0) < 0.98:
                    time.sleep(0.02)
                    obs = robot.get_observation()
                obs_frame = build_dataset_frame(ds_features, obs, OBS_STR)["observation.state"]
                obs_names = ds_features["observation.state"]["names"]
                assert obs_frame[obs_names.index("kRightHand.closure")] == pytest.approx(1.0, abs=0.02)
                assert obs_frame[obs_names.index("kLeftHand.closure")] == pytest.approx(0.0, abs=0.02)
            finally:
                robot.disconnect()

    @pytest.mark.parametrize("controller", [None, "GrootLocomotionController"])
    def test_first_recorded_action_matches_measured_pose(self, headhand_server, teleop, tmp_path, controller):
        mocks = _make_sdk_mocks(mode_machine=4)
        stub = _make_stub_controller() if controller else None
        patches, *_ = _make_patches(mocks, controller=stub)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(headhand_server, mocks, tmp_path, config_kwargs={"controller": controller})
            robot.connect(calibrate=False)
            try:
                ds_features = _dataset_features(robot)
                obs = robot.get_observation()
                teleop.send_feedback(obs)
                act = teleop.get_action()
                act_frame = build_dataset_frame(ds_features, act, ACTION)[ACTION]
                obs_frame = build_dataset_frame(ds_features, obs, OBS_STR)["observation.state"]
                act_names = ds_features[ACTION]["names"]
                obs_names = ds_features["observation.state"]["names"]
                for key in obs_names:
                    assert act_frame[act_names.index(key)] == pytest.approx(obs_frame[obs_names.index(key)])
                if controller:
                    assert act_names[: len(obs_names)] == obs_names
                assert any(abs(obs[key]) > 0.1 for key in ARM_KEYS)
            finally:
                robot.disconnect()
