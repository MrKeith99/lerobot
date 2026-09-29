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
    ALL_ACTION_KEYS,
    ARM_MODE_ACTION_KEYS,
    ARM_MODE_STATE_KEYS,
    CLOSURE_ACTION_KEYS,
    CLOSURE_ARM_MODE_ACTION_KEYS,
    CLOSURE_ARM_MODE_STATE_KEYS,
    G1_23_BODY_JOINTS,
    HAND_CLOSURE_KEYS,
    HEAD_HAND_MOTORS,
    closure_to_hand_q,
    hand_motor_names,
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


_MODE_KEYS = {
    "closure": (CLOSURE_ACTION_KEYS, CLOSURE_ARM_MODE_ACTION_KEYS),
    "per_motor": (ALL_ACTION_KEYS, ARM_MODE_ACTION_KEYS),
}
_CONTROLLER_STATE_KEYS = {
    "closure": CLOSURE_ARM_MODE_STATE_KEYS,
    "per_motor": ARM_MODE_STATE_KEYS,
}


class TestBaseHeightAction:
    def test_stock_default_off_g1ah_default_on(self):
        from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

        assert UnitreeG1Config().base_height_action is True
        assert UnitreeG1AhConfig().base_height_action is True

    @pytest.mark.parametrize(
        "controller_name, enabled, expected",
        [
            ("GrootLocomotionController", True, True),
            ("GrootLocomotionController", False, False),
            ("HolosomaLocomotionController", True, False),
        ],
    )
    def test_feature_only_with_groot_and_flag(
        self, headhand_server, tmp_path, controller_name, enabled, expected
    ):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks, controller=_make_stub_controller())
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(
                headhand_server,
                mocks,
                tmp_path,
                config_kwargs={"controller": controller_name, "base_height_action": enabled},
            )
            assert ("kBaseHeight.cmd" in robot.action_features) is expected
            assert "kBaseHeight.cmd" not in robot.observation_features

    def test_no_controller_has_no_height(self, g1ah_robot):
        robot, _ = g1ah_robot
        assert "kBaseHeight.cmd" not in robot.action_features

    @pytest.mark.parametrize("enabled", [True, False])
    def test_height_forwarded_to_controller_only_when_enabled(self, headhand_server, tmp_path, enabled):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks, controller=_make_stub_controller())
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(
                headhand_server,
                mocks,
                tmp_path,
                config_kwargs={"controller": "GrootLocomotionController", "base_height_action": enabled},
            )
            robot._update_controller_action({"kBaseHeight.cmd": 0.6, "remote.button.0": 1.0})
            assert ("kBaseHeight.cmd" in robot.controller_input) is enabled
            assert robot.controller_input["remote.button.0"] == 1.0


class TestG1AhLegacyCalibration:
    def test_legacy_motor_names_are_migrated_on_load(self, headhand_server, tmp_path):
        import json

        from lerobot.robots.unitree_g1_ah.g1_ah_joints import LEGACY_MOTOR_NAMES

        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(headhand_server, mocks, tmp_path)
            legacy = {old: robot.calibration[new] for old, new in LEGACY_MOTOR_NAMES.items()}
            robot.calibration = legacy
            robot._save_calibration()

            robot._load_calibration()

            assert set(robot.calibration) == set(HEAD_HAND_MOTORS)
            assert robot.calibration["kHeadYaw"] == legacy["xl330_joint"]
            assert robot.calibration["kLeftHandMotor11"] == legacy["left_hand_finger1_motor1"]
            on_disk = json.loads(robot.calibration_fpath.read_text())
            assert set(on_disk) == set(HEAD_HAND_MOTORS)


class TestG1AhFeatures:
    def test_default_hand_representation_is_closure(self, g1ah_robot):
        robot, _ = g1ah_robot
        assert robot.config.hand_representation == "closure"
        assert len(robot.observation_features) == 33
        assert len(robot.action_features) == 33

    @pytest.mark.parametrize("hand_representation", ["closure", "per_motor"])
    def test_observation_and_action_features_no_controller(
        self, headhand_server, tmp_path, hand_representation
    ):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(
                headhand_server, mocks, tmp_path, config_kwargs={"hand_representation": hand_representation}
            )
            state_keys, _ = _MODE_KEYS[hand_representation]
            assert list(robot.observation_features) == list(state_keys)
            assert list(robot.action_features) == list(state_keys)

    @pytest.mark.parametrize("base_height_action", [True, False])
    @pytest.mark.parametrize("hand_representation", ["closure", "per_motor"])
    def test_action_features_with_controller(
        self, headhand_server, tmp_path, hand_representation, base_height_action
    ):
        mocks = _make_sdk_mocks(mode_machine=4)
        controller = _make_stub_controller()
        patches, *_ = _make_g1ah(mocks, controller=controller)
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
                    "base_height_action": base_height_action,
                },
            )
            _, arm_mode_keys = _MODE_KEYS[hand_representation]
            state_keys = _CONTROLLER_STATE_KEYS[hand_representation]
            expected = list(arm_mode_keys) + (["kBaseHeight.cmd"] if base_height_action else [])
            assert list(robot.action_features) == expected
            assert list(robot.observation_features) == list(state_keys)
            assert expected[: len(state_keys)] == list(state_keys)


def _latest_goal_ticks(headhand_server, names, timeout_s=2.0):
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for cmd in reversed(headhand_server.received):
            goal_ticks = cmd.get("goal_ticks") or {}
            if all(name in goal_ticks for name in names):
                return goal_ticks
        time.sleep(0.01)
    return None


def _wait_for_observation(robot, key, expected, timeout_s=2.0):
    import time

    deadline = time.time() + timeout_s
    obs = robot.get_observation()
    while time.time() < deadline and abs(obs.get(key, -1.0) - expected) > 0.02:
        time.sleep(0.02)
        obs = robot.get_observation()
    return obs


class TestG1AhHandClosure:
    @pytest.mark.parametrize("closure", [0.0, 0.5, 1.0])
    def test_closure_action_sends_interpolated_hand_ticks(self, g1ah_robot, headhand_server, closure):
        from lerobot.robots.unitree_g1_ah.g1_ah_devices import rad_to_ticks

        robot, _ = g1ah_robot
        robot.connect(calibrate=False)
        headhand_server.received.clear()
        sent = robot.send_action({"kRightHand.closure": closure})

        names = hand_motor_names("right")
        goal_ticks = _latest_goal_ticks(headhand_server, names)
        assert goal_ticks is not None
        expected = [
            rad_to_ticks("scs0009", q, robot.calibration[name])
            for name, q in zip(names, closure_to_hand_q("right", closure), strict=True)
        ]
        assert [goal_ticks[name] for name in names] == expected
        assert not any(name in goal_ticks for name in hand_motor_names("left"))
        assert sent["kRightHand.closure"] == pytest.approx(closure)

    def test_closure_is_clipped(self, g1ah_robot):
        robot, _ = g1ah_robot
        robot.connect(calibrate=False)
        sent = robot.send_action({"kLeftHand.closure": 1.7, "kRightHand.closure": -0.3})
        assert sent["kLeftHand.closure"] == 1.0
        assert sent["kRightHand.closure"] == 0.0
        assert [sent[f"{name}.q"] for name in hand_motor_names("left")] == pytest.approx(
            closure_to_hand_q("left", 1.0), abs=1e-6
        )

    def test_closure_overrides_per_motor_keys(self, g1ah_robot):
        robot, _ = g1ah_robot
        robot.connect(calibrate=False)
        action = dict.fromkeys((f"{name}.q" for name in hand_motor_names("right")), 0.0)
        action["kRightHand.closure"] = 1.0
        sent = robot.send_action(action)
        assert [sent[f"{name}.q"] for name in hand_motor_names("right")] == pytest.approx(
            closure_to_hand_q("right", 1.0), abs=1e-6
        )

    @pytest.mark.parametrize("closure", [0.0, 0.3, 1.0])
    def test_observation_reports_measured_closure(self, g1ah_robot, closure):
        robot, _ = g1ah_robot
        robot.connect(calibrate=False)
        robot.send_action(dict.fromkeys(HAND_CLOSURE_KEYS, closure))
        obs = _wait_for_observation(robot, "kLeftHand.closure", closure)
        for key in HAND_CLOSURE_KEYS:
            assert obs[key] == pytest.approx(closure, abs=0.02)
            assert 0.0 <= obs[key] <= 1.0

    def test_per_motor_mode_ignores_closure_keys(self, headhand_server, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            robot = _new_robot(
                headhand_server, mocks, tmp_path, config_kwargs={"hand_representation": "per_motor"}
            )
            robot.connect(calibrate=False)
            try:
                sent = robot.send_action({"kRightHand.closure": 1.0})
                assert not any(f"{name}.q" in sent for name in hand_motor_names("right"))
                assert not any(key in robot.get_observation() for key in HAND_CLOSURE_KEYS)
            finally:
                robot.disconnect()


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

    def test_missing_body_slots_are_recorded_as_zero(self, g1ah_robot):
        from lerobot.robots.unitree_g1_ah.g1_ah_joints import BODY_KEYS, INVALID_BODY_KEYS

        robot, _ = g1ah_robot
        robot.connect(calibrate=False)
        obs = robot.get_observation()
        assert list(robot.observation_features)[:29] == list(BODY_KEYS)
        for key in INVALID_BODY_KEYS:
            assert obs[key] == 0.0
        assert obs["kLeftElbow.q"] != 0.0

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
        action["kHeadYaw.q"] = 5.0
        robot.send_action(action)

        import time

        deadline = time.time() + 2.0
        found = None
        while time.time() < deadline and found is None:
            for cmd in headhand_server.received:
                goal_ticks = cmd.get("goal_ticks")
                if goal_ticks and "kHeadYaw" in goal_ticks:
                    found = goal_ticks["kHeadYaw"]
                    break
            time.sleep(0.01)

        assert found is not None
        calib = robot.calibration["kHeadYaw"]
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


class TestG1AhHeadhandIp:
    def test_sim_mode_defaults_to_localhost(self, headhand_server, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from lerobot.robots.unitree_g1_ah.unitree_g1_ah import UnitreeG1Ah

            cfg = UnitreeG1AhConfig(
                robot_ip="192.168.123.164",
                is_simulation=True,
                headhand_state_port=headhand_server.state_port,
                headhand_cmd_port=headhand_server.cmd_port,
                calibration_dir=tmp_path,
                id="test",
            )
            robot = UnitreeG1Ah(cfg)
            assert robot.headhand.ip == "127.0.0.1"

    def test_real_mode_defaults_to_robot_ip(self, headhand_server, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from lerobot.robots.unitree_g1_ah.unitree_g1_ah import UnitreeG1Ah

            cfg = UnitreeG1AhConfig(
                robot_ip="192.168.123.164",
                is_simulation=False,
                headhand_state_port=headhand_server.state_port,
                headhand_cmd_port=headhand_server.cmd_port,
                calibration_dir=tmp_path,
                id="test",
            )
            robot = UnitreeG1Ah(cfg)
            assert robot.headhand.ip == "192.168.123.164"

    def test_headhand_ip_override_wins_in_simulation(self, headhand_server, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from lerobot.robots.unitree_g1_ah.unitree_g1_ah import UnitreeG1Ah

            cfg = UnitreeG1AhConfig(
                robot_ip="192.168.123.164",
                is_simulation=True,
                headhand_ip="10.0.0.5",
                headhand_state_port=headhand_server.state_port,
                headhand_cmd_port=headhand_server.cmd_port,
                calibration_dir=tmp_path,
                id="test",
            )
            robot = UnitreeG1Ah(cfg)
            assert robot.headhand.ip == "10.0.0.5"


class TestG1AhSimulationConnect:
    def test_sim_connect_with_empty_calibration_writes_defaults(self, headhand_server, tmp_path):
        mocks = _make_sdk_mocks(mode_machine=4)
        patches, *_ = _make_g1ah(mocks)

        fake_inner_env = MagicMock()
        fake_inner_env.simulator = None  # no elastic band / bridge joystick to poll
        fake_env_wrapper = {"hub_env": {0: MagicMock(envs=[fake_inner_env])}}

        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "lerobot.robots.unitree_g1.unitree_g1._SDKChannelFactoryInitialize",
                    MagicMock(),
                )
            )
            stack.enter_context(
                patch(
                    "lerobot.robots.unitree_g1.unitree_g1._SDKChannelPublisher",
                    MagicMock(return_value=mocks["publisher_mock"]),
                )
            )
            stack.enter_context(
                patch(
                    "lerobot.robots.unitree_g1.unitree_g1._SDKChannelSubscriber",
                    MagicMock(return_value=mocks["subscriber_mock"]),
                )
            )
            stack.enter_context(patch("lerobot.envs.make_env", return_value=fake_env_wrapper))

            from lerobot.robots.unitree_g1_ah.unitree_g1_ah import UnitreeG1Ah

            cfg = UnitreeG1AhConfig(
                robot_ip="127.0.0.1",
                is_simulation=True,
                headhand_state_port=headhand_server.state_port,
                headhand_cmd_port=headhand_server.cmd_port,
                calibration_dir=tmp_path,
                id="test",
            )
            robot = UnitreeG1Ah(cfg)
            assert robot.calibration == {}

            robot.connect(calibrate=True)
            try:
                assert robot.is_calibrated
                for name in HEAD_HAND_MOTORS:
                    assert robot.calibration[name] == default_calibration(name)
                assert robot.calibration_fpath.is_file()
            finally:
                robot.disconnect()
