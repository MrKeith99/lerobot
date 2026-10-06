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

"""Tests for Unitree G1 robot. Meant to be run in an environment where the Unitree SDK is installed."""

import contextlib
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.utils.import_utils import _unitree_sdk_available

if not _unitree_sdk_available:
    pytest.skip("Unitree SDK not available", allow_module_level=True)

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import (
    LOCOMOTION_TOGGLE_KEY,
    NUM_MOTORS,
    REMOTE_AXES,
    REMOTE_BUTTONS,
    REMOTE_KEYS,
    SIM_RESET_KEY,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    default_remote_input,
    get_gravity_orientation,
)

# ---------------------------------------------------------------------------
# Unit tests for g1_utils (no SDK needed)
# ---------------------------------------------------------------------------


class TestG1Utils:
    def test_num_motors(self):
        assert NUM_MOTORS == 29

    def test_joint_index_count(self):
        assert len(G1_29_JointIndex) == 29

    def test_joint_arm_index_count(self):
        assert len(G1_29_JointArmIndex) == 14

    def test_arm_indices_are_subset_of_full(self):
        full_values = {j.value for j in G1_29_JointIndex}
        arm_values = {j.value for j in G1_29_JointArmIndex}
        assert arm_values.issubset(full_values)

    def test_arm_indices_start_at_15(self):
        assert min(j.value for j in G1_29_JointArmIndex) == 15
        assert max(j.value for j in G1_29_JointArmIndex) == 28

    def test_enum_naming_consistency(self):
        """Verify all wrist joints use consistent PascalCase naming."""
        wrist_joints = [j for j in G1_29_JointIndex if "Wrist" in j.name]
        for j in wrist_joints:
            # Should be "WristYaw", "WristPitch", "WristRoll" — no lowercase after "Wrist"
            after_wrist = j.name.split("Wrist")[1]
            assert after_wrist[0].isupper(), f"{j.name} has inconsistent casing after 'Wrist'"

    def test_remote_keys_structure(self):
        assert len(REMOTE_AXES) == 4
        assert len(REMOTE_BUTTONS) == 16
        assert len(REMOTE_KEYS) == 20
        assert REMOTE_KEYS == REMOTE_AXES + REMOTE_BUTTONS

    def test_default_remote_input(self):
        d = default_remote_input()
        assert len(d) == 20
        assert all(v == 0.0 for v in d.values())
        assert set(d.keys()) == set(REMOTE_KEYS)

    def test_gravity_orientation_identity(self):
        """Quaternion [1, 0, 0, 0] (no rotation) should give gravity along -z."""
        g = get_gravity_orientation([1.0, 0.0, 0.0, 0.0])
        assert g.shape == (3,)
        assert g.dtype == np.float32
        np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-6)

    def test_gravity_orientation_dtype(self):
        g = get_gravity_orientation(np.array([1.0, 0.0, 0.0, 0.0]))
        assert g.dtype == np.float32


# ---------------------------------------------------------------------------
# Unit tests for UnitreeG1Config (no SDK needed)
# ---------------------------------------------------------------------------


class TestUnitreeG1Config:
    def test_default_config(self):
        cfg = UnitreeG1Config()
        assert len(cfg.kp) == 29
        assert len(cfg.kd) == 29
        assert len(cfg.default_positions) == 29
        assert cfg.is_simulation is True
        assert cfg.controller is None
        assert cfg.gravity_compensation is False

    def test_gains_are_positive(self):
        cfg = UnitreeG1Config()
        assert all(v > 0 for v in cfg.kp)
        assert all(v > 0 for v in cfg.kd)

    def test_config_copies_gains(self):
        """Each config instance should have its own copy of gains."""
        cfg1 = UnitreeG1Config()
        cfg2 = UnitreeG1Config()
        cfg1.kp[0] = 999.0
        assert cfg2.kp[0] != 999.0

    def test_default_sim_env_repo_id(self):
        assert UnitreeG1Config().sim_env_repo_id == "k-valentin/unitree-g1-mujoco"

    def test_sim_world_defaults_to_the_sims_own(self):
        cfg = UnitreeG1Config()
        assert cfg.sim_world is None
        assert cfg.sim_world_randomize is None

    def test_negative_sim_world_randomize_is_rejected(self):
        with pytest.raises(ValueError, match="sim_world_randomize"):
            UnitreeG1Config(sim_world_randomize=-0.01)


# ---------------------------------------------------------------------------
# Robot mock and integration tests
# ---------------------------------------------------------------------------


def _make_lowstate_msg_mock():
    """Create a mock that mimics the SDK LowState_ message."""
    msg = MagicMock()
    for i in range(29):
        motor = MagicMock()
        motor.q = float(i) * 0.1
        motor.dq = float(i) * 0.01
        motor.tau_est = float(i) * 0.001
        motor.temperature = 30.0 + i
        msg.motor_state.__getitem__ = lambda self, idx, _motors={}: _motors.setdefault(
            idx, MagicMock(q=idx * 0.1, dq=idx * 0.01, tau_est=idx * 0.001, temperature=30.0 + idx)
        )

    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    msg.imu_state.gyroscope = [0.1, 0.2, 0.3]
    msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
    msg.imu_state.rpy = [0.0, 0.0, 0.0]
    msg.imu_state.temperature = 25.0
    msg.wireless_remote = b"\x00" * 40
    msg.mode_machine = 0
    return msg


def _make_sdk_mocks():
    """Create mocks for the Unitree SDK modules used by UnitreeG1."""
    lowcmd_default = MagicMock()
    lowcmd_default.mode_pr = 0
    lowcmd_default.motor_cmd = [MagicMock() for _ in range(35)]

    crc_mock = MagicMock()
    crc_mock.Crc.return_value = 0

    lowstate_msg = _make_lowstate_msg_mock()

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


@pytest.fixture
def unitree_g1():
    """Create a UnitreeG1 robot with all SDK dependencies mocked."""
    with _mocked_unitree_g1() as (robot, mocks):
        yield robot, mocks


@contextlib.contextmanager
def _mocked_unitree_g1(controller=None, **config_kwargs):
    mocks = _make_sdk_mocks()

    mock_channel_init = MagicMock()
    mock_channel_pub = MagicMock(return_value=mocks["publisher_mock"])
    mock_channel_sub = MagicMock(return_value=mocks["subscriber_mock"])

    with (
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.make_cameras_from_configs",
            return_value={},
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.G1_29_ArmIK",
            return_value=MagicMock(),
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1._SDKChannelFactoryInitialize",
            mock_channel_init,
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1._SDKChannelPublisher",
            mock_channel_pub,
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1._SDKChannelSubscriber",
            mock_channel_sub,
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.unitree_hg_msg_dds__LowCmd_",
            MagicMock(return_value=mocks["lowcmd_default"]),
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.hg_LowCmd",
            MagicMock,
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.hg_LowState",
            MagicMock,
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.CRC",
            MagicMock(return_value=mocks["crc_mock"]),
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.make_locomotion_controller",
            return_value=MagicMock(control_dt=0.02) if controller else None,
        ),
    ):
        from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

        cfg = UnitreeG1Config(
            is_simulation=True, gravity_compensation=False, controller=controller, **config_kwargs
        )
        robot = UnitreeG1(cfg)
        try:
            yield robot, mocks
        finally:
            if robot.is_connected:
                robot.disconnect()


def test_init_state(unitree_g1):
    robot, _ = unitree_g1
    assert not robot.is_connected
    assert robot.controller is None


def test_observation_features(unitree_g1):
    robot, _ = unitree_g1
    features = robot.observation_features
    # Should have .q for all 29 joints (no cameras configured)
    assert len(features) == 29
    for joint in G1_29_JointIndex:
        assert f"{joint.name}.q" in features


def test_action_features_no_controller(unitree_g1):
    robot, _ = unitree_g1
    features = robot.action_features
    # Without controller: all 29 joints
    assert len(features) == 29
    for joint in G1_29_JointIndex:
        assert f"{joint.name}.q" in features


_LOWER_BODY_KEYS = {f"{j.name}.q" for j in G1_29_JointIndex if j.value < 15}
_ARM_KEYS = [f"{j.name}.q" for j in G1_29_JointIndex if j.value >= 15]
_NAV = ["kNavVx.cmd", "kNavVy.cmd", "kNavYawRate.cmd"]


@pytest.mark.parametrize(
    "controller, config_kwargs, expected_extra",
    [
        ("GrootLocomotionController", {}, [*_NAV, "kBaseHeight.cmd"]),
        ("GrootLocomotionController", {"base_height_action": False}, _NAV),
        ("HolosomaLocomotionController", {}, _NAV),
    ],
)
def test_controller_mode_features_drop_legs_and_waist(controller, config_kwargs, expected_extra):
    with _mocked_unitree_g1(controller=controller, **config_kwargs) as (robot, _):
        assert list(robot.observation_features) == _ARM_KEYS
        assert list(robot.action_features) == _ARM_KEYS + expected_extra
        assert not _LOWER_BODY_KEYS & set(robot.action_features)
        assert not any(key.startswith("remote.") for key in robot.action_features)


def test_controller_mode_forwards_nav_and_height_to_controller():
    with _mocked_unitree_g1(controller="GrootLocomotionController") as (robot, _):
        robot._update_controller_action(
            {"kNavVx.cmd": 0.4, "kNavVy.cmd": -0.1, "kNavYawRate.cmd": 0.3, "kBaseHeight.cmd": 0.6}
        )
        assert [robot.controller_input[key] for key in _NAV] == [0.4, -0.1, 0.3]
        assert robot.controller_input["kBaseHeight.cmd"] == 0.6


def test_get_observation_before_connect(unitree_g1):
    robot, _ = unitree_g1
    obs = robot.get_observation()
    assert obs == {}


def test_disconnect_idempotent(unitree_g1):
    robot, _ = unitree_g1
    # Should not raise even when not connected
    robot.disconnect()


def test_connect_uses_configured_sim_env_repo_id(unitree_g1):
    robot, _ = unitree_g1
    fake_inner_env = MagicMock()
    fake_inner_env.simulator = None  # no elastic band / bridge joystick to poll
    fake_env_wrapper = {"hub_env": {0: MagicMock(envs=[fake_inner_env])}}

    embodiment_vars = tuple(
        f"UNITREE_G1_MUJOCO_{name}" for name in ("BODY", "END_EFFECTOR", "HEAD_MOUNT", "HEAD_SENSOR")
    )
    seen_env = {}

    def fake_make_env(*args, **kwargs):
        seen_env.update({key: os.environ.get(key) for key in embodiment_vars})
        return fake_env_wrapper

    with patch("lerobot.envs.make_env", side_effect=fake_make_env) as mock_make_env:
        robot.connect()
        mock_make_env.assert_called_once_with(robot.config.sim_env_repo_id, trust_remote_code=True)
        assert robot.sim_env is fake_inner_env
    # The sim gets the embodiment through its environment variables
    assert seen_env == dict(zip(embodiment_vars, ("29dof", "rubber_hand", "fixed", "d435i"), strict=True))


def _connect_and_capture_world_vars(robot):
    fake_inner_env = MagicMock()
    fake_inner_env.simulator = None
    fake_env_wrapper = {"hub_env": {0: MagicMock(envs=[fake_inner_env])}}
    names = ("UNITREE_G1_MUJOCO_WORLD", "UNITREE_G1_MUJOCO_WORLD_RANDOMIZE")
    seen_env = {}

    def fake_make_env(*args, **kwargs):
        seen_env.update({key: os.environ.get(key) for key in names})
        return fake_env_wrapper

    with patch("lerobot.envs.make_env", side_effect=fake_make_env):
        robot.connect()
    return seen_env, names


def test_connect_exports_the_sim_world_when_set(unitree_g1):
    robot, _ = unitree_g1
    robot.config.sim_world = "pick_cylinder"
    robot.config.sim_world_randomize = 0.02
    seen_env, names = _connect_and_capture_world_vars(robot)
    assert seen_env == dict(zip(names, ("pick_cylinder", "0.02"), strict=True))
    assert all(name not in os.environ for name in names)


def test_connect_exports_only_the_world_that_is_set(unitree_g1):
    robot, _ = unitree_g1
    robot.config.sim_world = "pick_cylinder"
    seen_env, names = _connect_and_capture_world_vars(robot)
    assert seen_env == {names[0]: "pick_cylinder", names[1]: None}


def test_connect_does_not_export_the_world_when_unset(unitree_g1):
    robot, _ = unitree_g1
    seen_env, names = _connect_and_capture_world_vars(robot)
    assert seen_env == dict.fromkeys(names)


def test_disconnect_closes_sim_without_image_publisher(unitree_g1, caplog):
    """With image publishing off the sim has image_publish_process=None; disconnect still closes it."""
    robot, _ = unitree_g1
    sim_env = MagicMock()
    sim_env.simulator.sim_env.image_publish_process = None
    robot.sim_env = sim_env
    with caplog.at_level("WARNING"):
        robot.disconnect()
    sim_env.close.assert_called_once()
    assert "Error closing sim_env" not in caplog.text
    assert robot.sim_env is None


def _attach_fake_sim(robot, pressed: set[int], band_enabled: bool = False, num_buttons: int = 13):
    """Give the robot a fake MuJoCo env: elastic band, bridge low_cmd and a joystick with `pressed` buttons."""
    band = MagicMock()
    band.enable = band_enabled
    band.length = 5.0
    joystick = MagicMock()
    joystick.get_numbuttons.return_value = num_buttons
    joystick.get_button.side_effect = lambda index: index in pressed
    bridge = MagicMock()
    bridge.joystick = joystick
    bridge.low_cmd.motor_cmd = [MagicMock() for _ in range(35)]
    sim = MagicMock()
    sim.elastic_band = band
    sim.unitree_bridge = bridge
    robot.sim_env = MagicMock()
    robot.sim_env.simulator.sim_env = sim
    return band, bridge


def test_sim_band_attached_false_without_sim(unitree_g1):
    robot, _ = unitree_g1
    assert robot.sim_env is None
    assert robot._sim_band_attached() is False
    robot._poll_sim_gamepad_buttons()  # no sim: must be a no-op


def test_sim_band_toggle_button_is_edge_triggered(unitree_g1):
    robot, _ = unitree_g1
    pressed: set[int] = set()
    band, _ = _attach_fake_sim(robot, pressed, band_enabled=True)
    toggle = robot.config.sim_band_toggle_button

    robot._poll_sim_gamepad_buttons()
    assert band.enable is True

    pressed.add(toggle)
    robot._poll_sim_gamepad_buttons()
    assert band.enable is False
    assert robot._sim_band_attached() is False
    robot._poll_sim_gamepad_buttons()  # still held: no second toggle
    assert band.enable is False

    pressed.discard(toggle)
    robot._poll_sim_gamepad_buttons()
    pressed.add(toggle)
    robot._poll_sim_gamepad_buttons()
    assert band.enable is True
    assert robot._sim_band_attached() is True


def test_sim_band_toggle_button_out_of_range_is_ignored(unitree_g1):
    robot, _ = unitree_g1
    band, _ = _attach_fake_sim(robot, {robot.config.sim_band_toggle_button}, band_enabled=True, num_buttons=5)
    robot._poll_sim_gamepad_buttons()
    assert band.enable is True


def test_sim_reset_button_reattaches_band_and_limps_legs(unitree_g1):
    robot, mocks = unitree_g1
    robot.msg = mocks["lowcmd_default"]
    band, bridge = _attach_fake_sim(robot, {robot.config.sim_reset_button}, band_enabled=False)

    robot._poll_sim_gamepad_buttons()

    assert band.enable is True
    assert band.length == 0
    robot.sim_env.reset.assert_called_once_with()
    for motor in range(15):
        assert robot.msg.motor_cmd[motor].kp == 0.0
        assert robot.msg.motor_cmd[motor].kd == 0.0
        assert robot.msg.motor_cmd[motor].tau == 0.0
        assert bridge.low_cmd.motor_cmd[motor].kp == 0.0
        assert bridge.low_cmd.motor_cmd[motor].kd == 0.0
    assert robot.msg.motor_cmd[15].kp != 0.0  # arms untouched
    assert bridge.low_cmd.motor_cmd[15].kp != 0.0


def _band_action(value: float | None) -> dict[str, float]:
    return {} if value is None else {LOCOMOTION_TOGGLE_KEY: value}


def test_sim_band_toggle_key_rising_edge_works_without_joystick(unitree_g1):
    robot, _ = unitree_g1
    band, bridge = _attach_fake_sim(robot, set(), band_enabled=True)
    bridge.joystick = None

    robot._request_sim_band_toggle(_band_action(0.0))
    robot._poll_sim_gamepad_buttons()
    assert band.enable is True

    robot._request_sim_band_toggle(_band_action(1.0))
    robot._poll_sim_gamepad_buttons()
    assert band.enable is False

    robot._request_sim_band_toggle(_band_action(0.0))
    robot._request_sim_band_toggle(_band_action(1.0))
    robot._poll_sim_gamepad_buttons()
    assert band.enable is True


def test_sim_band_toggle_key_does_not_repeat_while_held(unitree_g1):
    robot, _ = unitree_g1
    band, _ = _attach_fake_sim(robot, set(), band_enabled=True)

    for _ in range(3):
        robot._request_sim_band_toggle(_band_action(1.0))
        robot._poll_sim_gamepad_buttons()
    assert band.enable is False

    robot._request_sim_band_toggle(_band_action(None))
    robot._request_sim_band_toggle(_band_action(1.0))
    robot._poll_sim_gamepad_buttons()
    assert band.enable is True


def test_sim_band_toggle_key_ignored_without_toggle_button():
    with _mocked_unitree_g1(sim_band_toggle_button=None) as (robot, _):
        band, _ = _attach_fake_sim(robot, set(), band_enabled=True)
        robot._request_sim_band_toggle(_band_action(1.0))
        robot._poll_sim_gamepad_buttons()
        assert band.enable is True


def test_sim_band_toggle_key_ignored_on_real_robot(unitree_g1):
    robot, _ = unitree_g1
    robot.config.is_simulation = False
    band, _ = _attach_fake_sim(robot, set(), band_enabled=True)
    robot._request_sim_band_toggle(_band_action(1.0))
    robot._poll_sim_gamepad_buttons()
    assert band.enable is True


def test_send_action_consumes_band_toggle_key_and_does_not_forward_it(unitree_g1):
    robot, _ = unitree_g1
    band, _ = _attach_fake_sim(robot, set(), band_enabled=True)
    robot.publish_lowcmd = MagicMock()
    sent = robot.send_action({LOCOMOTION_TOGGLE_KEY: 1.0})
    assert LOCOMOTION_TOGGLE_KEY not in sent
    assert LOCOMOTION_TOGGLE_KEY not in robot.action_features
    robot._poll_sim_gamepad_buttons()
    assert band.enable is False


def _reset_action(value: float | None) -> dict[str, float]:
    return {} if value is None else {SIM_RESET_KEY: value}


def test_sim_reset_key_rising_edge_works_without_joystick(unitree_g1):
    robot, mocks = unitree_g1
    robot.msg = mocks["lowcmd_default"]
    band, bridge = _attach_fake_sim(robot, set(), band_enabled=False)
    bridge.joystick = None

    robot._request_sim_reset(_reset_action(0.0))
    robot._poll_sim_gamepad_buttons()
    robot.sim_env.reset.assert_not_called()

    robot._request_sim_reset(_reset_action(1.0))
    robot._poll_sim_gamepad_buttons()
    robot.sim_env.reset.assert_called_once_with()
    assert band.enable is True
    assert band.length == 0
    assert robot.msg.motor_cmd[0].kp == 0.0
    assert bridge.low_cmd.motor_cmd[0].kp == 0.0


def test_sim_reset_key_does_not_repeat_while_held(unitree_g1):
    robot, _ = unitree_g1
    _attach_fake_sim(robot, set(), band_enabled=False)

    for _ in range(3):
        robot._request_sim_reset(_reset_action(1.0))
        robot._poll_sim_gamepad_buttons()
    assert robot.sim_env.reset.call_count == 1

    robot._request_sim_reset(_reset_action(None))
    robot._request_sim_reset(_reset_action(1.0))
    robot._poll_sim_gamepad_buttons()
    assert robot.sim_env.reset.call_count == 2


def test_sim_reset_key_ignored_without_reset_button():
    with _mocked_unitree_g1(sim_reset_button=None) as (robot, _):
        _attach_fake_sim(robot, set(), band_enabled=False)
        robot._request_sim_reset(_reset_action(1.0))
        robot._poll_sim_gamepad_buttons()
        robot.sim_env.reset.assert_not_called()


def test_sim_reset_key_ignored_on_real_robot(unitree_g1):
    robot, _ = unitree_g1
    robot.config.is_simulation = False
    _attach_fake_sim(robot, set(), band_enabled=False)
    robot._request_sim_reset(_reset_action(1.0))
    robot._poll_sim_gamepad_buttons()
    robot.sim_env.reset.assert_not_called()


def test_send_action_consumes_reset_key_and_does_not_forward_it(unitree_g1):
    robot, _ = unitree_g1
    _attach_fake_sim(robot, set(), band_enabled=False)
    robot.publish_lowcmd = MagicMock()
    sent = robot.send_action({SIM_RESET_KEY: 1.0})
    assert SIM_RESET_KEY not in sent
    assert SIM_RESET_KEY not in robot.action_features
    robot._poll_sim_gamepad_buttons()
    robot.sim_env.reset.assert_called_once_with()


class _FakeController:
    control_dt = 0.02

    def __init__(self, with_gains=True):
        self.default_angles = np.linspace(-0.5, 0.5, 29).astype(np.float32)
        if with_gains:
            self.kp = np.arange(29, dtype=np.float32) + 100.0
            self.kd = np.arange(29, dtype=np.float32) + 10.0
        self.calls = []
        self.run_step = MagicMock(side_effect=lambda *a: self.calls.append("run_step") or {})
        self.reset = MagicMock(side_effect=lambda: self.calls.append("reset"))


def _lowstate(q=0.0):
    state = MagicMock()
    for i in range(29):
        state.motor_state[i].q = q + 0.01 * i
    return state


@contextlib.contextmanager
def _real_gated_robot(gate=True, controller=None, **config_kwargs):
    with _mocked_unitree_g1(
        controller="GrootLocomotionController", locomotion_start_gate=gate, **config_kwargs
    ) as (robot, _):
        robot.config.is_simulation = False
        robot.controller = controller or _FakeController()
        robot.kp = np.array(robot.config.kp, dtype=np.float32)
        robot.kd = np.array(robot.config.kd, dtype=np.float32)
        robot.publish_lowcmd = MagicMock()
        yield robot


def _published_q(robot):
    action = robot.publish_lowcmd.call_args.args[0]
    return np.array([action[f"{G1_29_JointIndex(i).name}.q"] for i in range(15)])


def _toggle(robot):
    robot._request_locomotion_toggle({LOCOMOTION_TOGGLE_KEY: 0.0})
    robot._request_locomotion_toggle({LOCOMOTION_TOGGLE_KEY: 1.0})


def test_stand_ramp_must_be_positive():
    with pytest.raises(ValueError, match="stand_ramp_s"):
        UnitreeG1Config(stand_ramp_s=0.0)
    assert UnitreeG1Config().locomotion_start_gate is True
    assert UnitreeG1Config().stand_ramp_s == 3.0


def test_gate_does_not_run_policy_before_enable():
    with _real_gated_robot() as robot:
        for t in (0.0, 1.0, 5.0, 10.0):
            robot._controller_iteration(_lowstate(), t)
        robot.controller.run_step.assert_not_called()
        assert robot._locomotion_enabled is False


def test_gate_ramps_from_measured_to_standing_pose_then_holds():
    with _real_gated_robot() as robot:
        start = _lowstate(0.3)
        robot._controller_iteration(start, 100.0)
        np.testing.assert_allclose(
            _published_q(robot), [start.motor_state[i].q for i in range(15)], atol=1e-6
        )
        standing = robot.controller.default_angles[:15]
        measured = np.array([start.motor_state[i].q for i in range(15)])

        robot._controller_iteration(_lowstate(-1.0), 101.5)  # later measurements do not move the ramp start
        np.testing.assert_allclose(_published_q(robot), measured + (standing - measured) * 0.5, atol=1e-6)

        for t in (103.0, 110.0):
            robot._controller_iteration(_lowstate(0.9), t)
            np.testing.assert_allclose(_published_q(robot), standing, atol=1e-6)


def test_gate_uses_controller_gains_and_only_commands_legs_and_waist():
    with _real_gated_robot() as robot:
        robot._controller_iteration(_lowstate(), 0.0)
        action = robot.publish_lowcmd.call_args.args[0]
        assert sorted(action) == sorted(f"{G1_29_JointIndex(i).name}.q" for i in range(15))
        assert robot.publish_lowcmd.call_args.kwargs["kp"] is robot.controller.kp
        assert robot.publish_lowcmd.call_args.kwargs["kd"] is robot.controller.kd


def test_gate_falls_back_to_config_gains_and_default_positions():
    controller = _FakeController(with_gains=False)
    del controller.default_angles
    with _real_gated_robot(controller=controller) as robot:
        robot._controller_iteration(_lowstate(), 0.0)
        assert robot.publish_lowcmd.call_args.kwargs["kp"] is robot.kp
        assert robot.publish_lowcmd.call_args.kwargs["kd"] is robot.kd
        robot._controller_iteration(_lowstate(), 10.0)
        np.testing.assert_allclose(_published_q(robot), robot.config.default_positions[:15], atol=1e-6)


def test_gate_ignores_malformed_controller_standing_pose():
    controller = _FakeController()
    controller.default_angles = MagicMock()
    with _real_gated_robot(controller=controller) as robot:
        robot._controller_iteration(_lowstate(), 0.0)
        robot._controller_iteration(_lowstate(), 10.0)
        np.testing.assert_allclose(_published_q(robot), robot.config.default_positions[:15], atol=1e-6)


def test_gate_leaves_arm_motors_untouched():
    with _real_gated_robot() as robot:
        robot._controller_iteration(_lowstate(), 0.0)
        robot._controller_iteration(_lowstate(), 4.0)
        for call in robot.publish_lowcmd.call_args_list:
            assert all(G1_29_JointIndex[key[:-2]].value < 15 for key in call.args[0])


def test_enable_refused_before_ramp_ends():
    with _real_gated_robot() as robot:
        robot._controller_iteration(_lowstate(), 0.0)
        _toggle(robot)
        robot._controller_iteration(_lowstate(), 2.9)
        assert robot._locomotion_enabled is False
        robot.controller.run_step.assert_not_called()
        robot.controller.reset.assert_not_called()
        assert robot.ready_to_record() is False


def test_enable_resets_then_runs_policy_with_controller_gains():
    with _real_gated_robot() as robot:
        robot._controller_iteration(_lowstate(), 0.0)
        robot._controller_iteration(_lowstate(), 3.0)
        _toggle(robot)
        robot._controller_iteration(_lowstate(), 3.02)
        assert robot._locomotion_enabled is True
        assert robot.controller.calls == ["reset", "run_step"]
        assert robot.publish_lowcmd.call_args.kwargs["kp"] is robot.controller.kp
        robot._controller_iteration(_lowstate(), 3.04)
        assert robot.controller.calls == ["reset", "run_step", "run_step"]
        assert robot.ready_to_record() is True


def test_disable_holds_the_current_measured_pose_without_ramp():
    with _real_gated_robot() as robot:
        robot._controller_iteration(_lowstate(), 0.0)
        robot._controller_iteration(_lowstate(), 3.0)
        _toggle(robot)
        robot._controller_iteration(_lowstate(), 3.02)
        robot.controller.run_step.reset_mock()

        _toggle_off = _lowstate(0.2)
        robot._request_locomotion_toggle({LOCOMOTION_TOGGLE_KEY: 0.0})
        robot._request_locomotion_toggle({LOCOMOTION_TOGGLE_KEY: 1.0})
        robot._controller_iteration(_toggle_off, 5.0)
        assert robot._locomotion_enabled is False
        robot.controller.run_step.assert_not_called()
        held = [_toggle_off.motor_state[i].q for i in range(15)]
        np.testing.assert_allclose(_published_q(robot), held, atol=1e-6)
        assert robot.publish_lowcmd.call_args.kwargs["kp"] is robot.controller.kp

        robot._controller_iteration(_lowstate(0.8), 5.02)  # keeps holding the pose captured at the stop
        np.testing.assert_allclose(_published_q(robot), held, atol=1e-6)
        assert robot.ready_to_record() is False


def test_locomotion_toggle_is_rising_edge_only():
    with _real_gated_robot() as robot:
        robot._controller_iteration(_lowstate(), 0.0)
        robot._controller_iteration(_lowstate(), 3.0)
        for _ in range(5):
            robot._request_locomotion_toggle({LOCOMOTION_TOGGLE_KEY: 1.0})
        robot._controller_iteration(_lowstate(), 3.02)
        assert robot._locomotion_enabled is True
        for _ in range(5):
            robot._request_locomotion_toggle({LOCOMOTION_TOGGLE_KEY: 1.0})
        robot._controller_iteration(_lowstate(), 3.04)
        assert robot._locomotion_enabled is True


def test_toggle_key_is_consumed_by_send_action_on_the_real_robot():
    with _real_gated_robot() as robot:
        sent = robot.send_action({LOCOMOTION_TOGGLE_KEY: 1.0})
        assert LOCOMOTION_TOGGLE_KEY not in sent
        assert robot._locomotion_toggle_requested is True


def test_gate_off_runs_policy_from_the_first_iteration():
    with _real_gated_robot(gate=False) as robot:
        robot._controller_iteration(_lowstate(), 0.0)
        assert robot.controller.calls == ["run_step"]
        robot.publish_lowcmd.assert_called_once()
        _toggle(robot)
        assert robot._locomotion_toggle_requested is False


def test_sim_with_gate_default_runs_policy_and_ignores_locomotion_toggle():
    with _mocked_unitree_g1(controller="GrootLocomotionController") as (robot, _):
        robot.controller = _FakeController()
        robot.publish_lowcmd = MagicMock()
        robot._controller_iteration(_lowstate(), 0.0)
        assert robot.controller.calls == ["run_step"]
        _toggle(robot)
        assert robot._locomotion_toggle_requested is False


@pytest.mark.parametrize(
    ("gate", "controller", "enabled", "expected"),
    [
        (True, "GrootLocomotionController", False, False),
        (True, "GrootLocomotionController", True, True),
        (False, "GrootLocomotionController", False, True),
        (True, None, False, True),
    ],
)
def test_ready_to_record_real_robot(gate, controller, enabled, expected):
    with _mocked_unitree_g1(controller=controller, locomotion_start_gate=gate) as (robot, _):
        robot.config.is_simulation = False
        robot._locomotion_enabled = enabled
        assert robot.ready_to_record() is expected


@pytest.mark.parametrize(
    ("is_simulation", "controller", "band_enabled", "expected"),
    [
        (False, None, True, True),
        (True, None, True, True),
        (True, "GrootLocomotionController", True, False),
        (True, "GrootLocomotionController", False, True),
    ],
)
def test_ready_to_record(is_simulation, controller, band_enabled, expected):
    with _mocked_unitree_g1(controller=controller) as (robot, _):
        robot.config.is_simulation = is_simulation
        _attach_fake_sim(robot, set(), band_enabled=band_enabled)
        assert robot.ready_to_record() is expected
