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

"""Tests for the UnitreeG1Ah robot. Meant to be run in an environment where the Unitree SDK is installed."""

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from lerobot.utils.import_utils import _unitree_sdk_available

if not _unitree_sdk_available:
    pytest.skip("Unitree SDK not available", allow_module_level=True)

from lerobot.robots.unitree_g1_ah.config_unitree_g1_ah import UnitreeG1AhConfig
from lerobot.robots.unitree_g1_ah.g1_ah_devices import default_calibration
from lerobot.robots.unitree_g1_ah.g1_ah_joints import (
    ARM_MODE_ACTION_KEYS,
    G1_23_BODY_JOINTS,
    HEAD_HAND_MOTORS,
)
from tests.mocks.mock_unitree_g1_ah_server import MockHeadHandServer


def _make_lowstate_msg_mock(mode_machine: int = 4):
    msg = MagicMock()
    msg.motor_state.__getitem__ = lambda self, idx, _motors={}: _motors.setdefault(
        idx, MagicMock(q=idx * 0.1, dq=idx * 0.01, tau_est=idx * 0.001, temperature=30.0 + idx)
    )
    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    msg.imu_state.gyroscope = [0.1, 0.2, 0.3]
    msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
    msg.imu_state.rpy = [0.0, 0.0, 0.0]
    msg.imu_state.temperature = 25.0
    msg.wireless_remote = b"\x00" * 40
    msg.mode_machine = mode_machine
    return msg


def _make_sdk_mocks(mode_machine: int = 4):
    lowcmd_default = MagicMock()
    lowcmd_default.mode_pr = 0
    lowcmd_default.motor_cmd = [MagicMock() for _ in range(35)]
    for cmd in lowcmd_default.motor_cmd:
        cmd.kp = 0.0
        cmd.kd = 0.0

    crc_mock = MagicMock()
    crc_mock.Crc.return_value = 0

    lowstate_msg = _make_lowstate_msg_mock(mode_machine)

    subscriber_mock = MagicMock()
    subscriber_mock.Read.return_value = lowstate_msg

    publisher_mock = MagicMock()

    return {
        "lowcmd_default": lowcmd_default,
        "crc_mock": crc_mock,
        "subscriber_mock": subscriber_mock,
        "publisher_mock": publisher_mock,
        "lowstate_msg": lowstate_msg,
    }


def _make_stub_controller():
    controller = MagicMock()
    controller.control_dt = 0.02
    controller.run_step.return_value = {}
    controller.reset = MagicMock()
    controller.kp = [50.0] * 29
    controller.kd = [3.0] * 29
    return controller


def _make_g1ah(mocks, *, config_kwargs=None, controller=None):
    mock_channel_init = MagicMock()
    mock_channel_pub = MagicMock(return_value=mocks["publisher_mock"])
    mock_channel_sub = MagicMock(return_value=mocks["subscriber_mock"])

    patches = [
        patch("lerobot.robots.unitree_g1.unitree_g1.make_cameras_from_configs", return_value={}),
        patch("lerobot.robots.unitree_g1.unitree_g1.G1_29_ArmIK", return_value=MagicMock()),
        patch("lerobot.robots.unitree_g1.unitree_sdk2_socket.ChannelFactoryInitialize", mock_channel_init),
        patch("lerobot.robots.unitree_g1.unitree_sdk2_socket.ChannelPublisher", mock_channel_pub),
        patch("lerobot.robots.unitree_g1.unitree_sdk2_socket.ChannelSubscriber", mock_channel_sub),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.unitree_hg_msg_dds__LowCmd_",
            MagicMock(return_value=mocks["lowcmd_default"]),
        ),
        patch("lerobot.robots.unitree_g1.unitree_g1.hg_LowCmd", MagicMock),
        patch("lerobot.robots.unitree_g1.unitree_g1.hg_LowState", MagicMock),
        patch("lerobot.robots.unitree_g1.unitree_g1.CRC", MagicMock(return_value=mocks["crc_mock"])),
    ]
    if controller is not None:
        patches.append(
            patch("lerobot.robots.unitree_g1.unitree_g1.make_locomotion_controller", return_value=controller)
        )

    return patches, mock_channel_init, mock_channel_pub, mock_channel_sub


@pytest.fixture
def headhand_server():
    with MockHeadHandServer() as server:
        yield server


def _new_robot(headhand_server, mocks, tmp_path, *, config_kwargs=None, controller=None):
    from lerobot.robots.unitree_g1_ah.unitree_g1_ah import UnitreeG1Ah

    kwargs = dict(config_kwargs or {})
    cfg = UnitreeG1AhConfig(
        robot_ip="127.0.0.1",
        headhand_state_port=headhand_server.state_port,
        headhand_cmd_port=headhand_server.cmd_port,
        calibration_dir=tmp_path,
        id="test",
        **kwargs,
    )
    robot = UnitreeG1Ah(cfg)
    robot.calibration = {name: default_calibration(name) for name in HEAD_HAND_MOTORS}
    robot._save_calibration()
    return robot


@pytest.fixture
def g1ah_robot(headhand_server, tmp_path):
    mocks = _make_sdk_mocks(mode_machine=4)
    patches, *_ = _make_g1ah(mocks)
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        robot = _new_robot(headhand_server, mocks, tmp_path)
        yield robot, mocks
        if robot.is_connected:
            robot.disconnect()


class TestG1AhIdentity:
    def test_name_and_calibration_dir_rev_1_0(self, headhand_server, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(headhand_server, mocks, tmp_path, config_kwargs={"revision": "rev_1_0"})
            assert robot.name == "unitree_g1_23dof_ah8_d455_2dof_rev_1_0"
            assert robot.robot_type == "unitree_g1_23dof_ah8_d455_2dof_rev_1_0"
            assert str(tmp_path) in str(robot.calibration_dir) or robot.calibration_dir == tmp_path

    def test_name_and_calibration_dir_base(self, headhand_server, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=1)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(headhand_server, mocks, tmp_path, config_kwargs={"revision": "base"})
            assert robot.name == "unitree_g1_23dof_ah8_d455_2dof"


class TestG1AhFeatures:
    def test_observation_and_action_features_no_controller(self, g1ah_robot):
        robot, _ = g1ah_robot
        assert len(robot.observation_features) == 41
        assert len(robot.action_features) == 41
        assert list(robot.action_features) == list(robot._motors_ft)

    def test_action_features_with_controller(self, headhand_server, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        controller = _make_stub_controller()
        patches, *_ = _make_g1ah(mocks, controller=controller)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(
                headhand_server, mocks, tmp_path, config_kwargs={"controller": "GrootLocomotionController"}
            )
            assert len(robot.action_features) == 32
            assert list(robot.action_features) == list(ARM_MODE_ACTION_KEYS)


class TestG1AhConnect:
    def test_connect_sets_is_connected(self, g1ah_robot):
        robot, _ = g1ah_robot
        robot.connect(calibrate=False)
        assert robot.is_connected

    def test_get_observation_keys(self, g1ah_robot):
        robot, _ = g1ah_robot
        robot.connect(calibrate=False)
        import time

        time.sleep(0.05)
        obs = robot.get_observation()
        expected_motor_keys = set(robot.observation_features) - set(robot._cameras_ft)
        assert expected_motor_keys.issubset(obs.keys())
        for joint in G1_23_BODY_JOINTS:
            assert f"{joint.name}.q" in obs
        for name in HEAD_HAND_MOTORS:
            assert f"{name}.q" in obs

    def test_send_action_zeroes_invalid_slot_gains(self, g1ah_robot):
        robot, mocks = g1ah_robot
        robot.connect(calibrate=False)
        action = dict.fromkeys(robot.action_features, 0.0)
        robot.send_action(action)
        lowcmd = mocks["lowcmd_default"]
        assert lowcmd.motor_cmd[13].kp == 0
        assert lowcmd.motor_cmd[20].kp == 0
        assert lowcmd.motor_cmd[27].kp == 0

    def test_send_action_head_hand_goal_ticks(self, g1ah_robot, headhand_server):
        robot, _ = g1ah_robot
        robot.connect(calibrate=False)
        action = dict.fromkeys(robot.action_features, 0.0)
        action["xl330_joint.q"] = 5.0
        robot.send_action(action)

        import time

        deadline = time.time() + 2.0
        found = None
        while time.time() < deadline and found is None:
            for cmd in headhand_server.received:
                goal_ticks = cmd.get("goal_ticks")
                if goal_ticks and "xl330_joint" in goal_ticks:
                    found = goal_ticks["xl330_joint"]
                    break
            time.sleep(0.01)

        assert found is not None
        calib = robot.calibration["xl330_joint"]
        from lerobot.robots.unitree_g1_ah.g1_ah_devices import rad_to_ticks

        expected = rad_to_ticks("xl330-m288", 0.7, calib)
        assert found == expected

    def test_mode_machine_mismatch_raises(self, headhand_server, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=1)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(headhand_server, mocks, tmp_path, config_kwargs={"revision": "rev_1_0"})
            with pytest.raises(ValueError):
                robot.connect(calibrate=False)
            assert not robot.is_connected

    def test_disconnect_twice_ok(self, g1ah_robot):
        robot, _ = g1ah_robot
        robot.connect(calibrate=False)
        robot.disconnect()
        assert not robot.is_connected
        robot.disconnect()


class TestG1AhHeadhandTimeout:
    def test_headhand_timeout_raises(self, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from lerobot.robots.unitree_g1_ah.unitree_g1_ah import UnitreeG1Ah

            cfg = UnitreeG1AhConfig(
                robot_ip="127.0.0.1",
                headhand_state_port=1,
                headhand_cmd_port=2,
                headhand_timeout_s=0.2,
                calibration_dir=tmp_path,
                id="test",
            )
            robot = UnitreeG1Ah(cfg)
            robot.calibration = {name: default_calibration(name) for name in HEAD_HAND_MOTORS}
            robot._save_calibration()
            with pytest.raises(TimeoutError):
                robot.connect(calibrate=False)
