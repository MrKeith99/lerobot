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

"""Tests for the embodiment options of UnitreeG1Config (body, end effector, head) and the features
they give the robot. No hardware/SDK required."""

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.end_effectors import HAND_CLOSURE_KEYS, HAND_SPECS
from lerobot.robots.unitree_g1.g1_utils import (
    ARM_KEYS,
    BASE_HEIGHT_KEY,
    BODY_KEYS,
    G1_23_INVALID_SDK_SLOTS,
    G1_LEG_SLOTS,
    NAV_KEYS,
)
from lerobot.robots.unitree_g1.headhand_zmq import HEADHAND_CMD_PORT, HEADHAND_STATE_PORT
from lerobot.robots.unitree_g1.heads import DEFAULT_HEAD_Q, HEAD_KEYS
from lerobot.teleoperators.unitree_g1_gamepad.unitree_g1_gamepad import TELEOP_ACTION_KEYS
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

AH_EMBODIMENT = {"body": "23dof", "end_effector": "amazing_hand", "head": "d455_pan_tilt"}

VALID_EMBODIMENTS = [
    ({"body": "29dof", "end_effector": "rubber_hand", "head": "none"}, "unitree_g1_29dof"),
    ({"body": "23dof", "end_effector": "rubber_hand", "head": "none"}, "unitree_g1_23dof"),
    ({"body": "29dof", "end_effector": "none", "head": "none"}, "unitree_g1_29dof_no_hand"),
    ({"body": "29dof", "end_effector": "dex1", "head": "none"}, "unitree_g1_29dof_dex1"),
    ({"body": "23dof", "end_effector": "dex3", "head": "none"}, "unitree_g1_23dof_dex3"),
    ({"body": "29dof", "end_effector": "amazing_hand", "head": "none"}, "unitree_g1_29dof_amazing_hand"),
    (
        {"body": "29dof", "end_effector": "rubber_hand", "head": "d455_pan_tilt"},
        "unitree_g1_29dof_d455_pan_tilt",
    ),
    (AH_EMBODIMENT, "unitree_g1_23dof_amazing_hand_d455_pan_tilt"),
]


def _ah_config(**kwargs) -> UnitreeG1Config:
    return UnitreeG1Config(**AH_EMBODIMENT, is_simulation=False, **kwargs)


@contextlib.contextmanager
def _robot(controller=None, **config_kwargs):
    """A UnitreeG1 built (not connected) without the SDK, cameras or a real locomotion controller."""
    with (
        patch("lerobot.robots.unitree_g1.unitree_g1.require_package", return_value=None),
        patch("lerobot.robots.unitree_g1.unitree_g1.make_cameras_from_configs", return_value={}),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.make_locomotion_controller",
            return_value=MagicMock(control_dt=0.02) if controller else None,
        ),
    ):
        from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

        yield UnitreeG1(UnitreeG1Config(controller=controller, **config_kwargs))


class TestEmbodimentConfigDefaults:
    def test_defaults(self):
        cfg = UnitreeG1Config()
        assert (cfg.body, cfg.end_effector, cfg.head) == ("29dof", "rubber_hand", "none")
        assert cfg.revision == "rev_1_0"
        assert cfg.is_simulation is True
        assert cfg.headhand_state_port == HEADHAND_STATE_PORT
        assert cfg.headhand_cmd_port == HEADHAND_CMD_PORT
        assert HEADHAND_STATE_PORT == 6003
        assert HEADHAND_CMD_PORT == 6002

    def test_sim_env_repo_id(self):
        assert _ah_config().sim_env_repo_id == "k-valentin/unitree-g1-mujoco"

    def test_type(self):
        assert _ah_config().type == "unitree_g1"

    @pytest.mark.parametrize("embodiment, robot_type", VALID_EMBODIMENTS)
    def test_robot_type_per_embodiment(self, embodiment, robot_type):
        assert UnitreeG1Config(**embodiment).robot_type == robot_type

    def test_robot_type_ignores_revision(self):
        assert _ah_config(revision="base").robot_type == _ah_config(revision="rev_1_0").robot_type

    def test_invalid_revision_raises(self):
        with pytest.raises(ValueError, match="revision"):
            _ah_config(revision="not_a_revision")

    def test_hand_representation_defaults_to_closure(self):
        assert _ah_config().hand_representation == "closure"

    def test_hand_representation_per_motor_accepted(self):
        assert _ah_config(hand_representation="per_motor").hand_representation == "per_motor"

    def test_invalid_hand_representation_raises(self):
        with pytest.raises(ValueError, match="hand_representation"):
            _ah_config(hand_representation="joints")

    def test_headhand_motors_only_for_the_bridge_embodiment(self):
        assert len(_ah_config().headhand_motors) == 18
        assert UnitreeG1Config().headhand_motors == {}
        assert UnitreeG1Config(end_effector="dex3").headhand_motors == {}
        assert set(UnitreeG1Config(head="d455_pan_tilt").headhand_motors) == {"kHeadYaw", "kHeadPitch"}
        assert len(UnitreeG1Config(end_effector="amazing_hand").headhand_motors) == 16


class TestEmbodimentValidation:
    @pytest.mark.parametrize("body", ["29dof", "23dof"])
    @pytest.mark.parametrize("end_effector", ["rubber_hand", "none", "dex1", "dex3", "amazing_hand"])
    @pytest.mark.parametrize("head", ["none", "d455_pan_tilt"])
    def test_every_combination_is_valid(self, body, end_effector, head):
        UnitreeG1Config(body=body, end_effector=end_effector, head=head)

    @pytest.mark.parametrize(
        "embodiment, match",
        [
            ({"body": "31dof"}, "body"),
            ({"end_effector": "gripper"}, "end_effector"),
            ({"head": "fixed"}, "head"),
        ],
    )
    def test_invalid_values_raise(self, embodiment, match):
        with pytest.raises(ValueError, match=match):
            UnitreeG1Config(**embodiment)

    def test_bad_kp_length_raises(self):
        with pytest.raises(ValueError):
            _ah_config(kp=[1.0] * 28)

    def test_bad_head_default_positions_length_raises(self):
        with pytest.raises(ValueError):
            _ah_config(head_default_positions=[0.0])
        with pytest.raises(ValueError):
            UnitreeG1Config(head_default_positions=[0.0, 0.0])

    def test_bad_hand_default_positions_length_raises(self):
        with pytest.raises(ValueError):
            _ah_config(hand_default_positions=[0.0] * 15)
        with pytest.raises(ValueError):
            UnitreeG1Config(end_effector="dex3", hand_default_positions=[0.0] * 16)


class TestEmbodimentDefaultPositions:
    @pytest.mark.parametrize(
        "end_effector, body, expected_len",
        [("none", "29dof", 0), ("dex1", "29dof", 2), ("dex3", "29dof", 14), ("amazing_hand", "23dof", 16)],
    )
    def test_hand_default_positions_length_per_end_effector(self, end_effector, body, expected_len):
        head = "d455_pan_tilt" if end_effector == "amazing_hand" else "none"
        cfg = UnitreeG1Config(body=body, end_effector=end_effector, head=head)
        assert len(cfg.hand_default_positions) == expected_len

    @pytest.mark.parametrize("end_effector", ["dex1", "dex3"])
    def test_hand_default_positions_are_open_pose(self, end_effector):
        spec = HAND_SPECS[end_effector]
        cfg = UnitreeG1Config(end_effector=end_effector)
        assert cfg.hand_default_positions == [*spec.open_q["left"], *spec.open_q["right"]]

    def test_head_default_positions_per_head(self):
        assert _ah_config().head_default_positions == list(DEFAULT_HEAD_Q)
        assert UnitreeG1Config().head_default_positions == []


class TestEmbodimentGains:
    def test_invalid_slots_zeroed_for_23dof(self):
        cfg = _ah_config()
        for i in G1_23_INVALID_SDK_SLOTS:
            assert cfg.kp[i] == 0.0
            assert cfg.kd[i] == 0.0

    def test_invalid_slots_zeroed_for_bare_23dof(self):
        cfg = UnitreeG1Config(body="23dof")
        assert all(cfg.kp[i] == 0.0 and cfg.kd[i] == 0.0 for i in G1_23_INVALID_SDK_SLOTS)

    @pytest.mark.parametrize("end_effector", ["none", "dex1", "dex3"])
    def test_no_slots_zeroed_for_29dof(self, end_effector):
        cfg = UnitreeG1Config(body="29dof", end_effector=end_effector)
        assert all(kp > 0 for kp in cfg.kp)
        assert all(kd > 0 for kd in cfg.kd)

    def test_valid_slots_nonzero(self):
        cfg = _ah_config()
        valid_slots = [i for i in range(29) if i not in G1_23_INVALID_SDK_SLOTS]
        assert all(cfg.kp[i] > 0 for i in valid_slots)
        assert all(cfg.kd[i] > 0 for i in valid_slots)

    def test_freeze_legs_zeroes_leg_slots_only(self):
        cfg = _ah_config(freeze_legs=True, controller=None)
        for i in G1_LEG_SLOTS:
            assert cfg.kp[i] == 0.0
            assert cfg.kd[i] == 0.0
        non_leg_valid = [i for i in range(29) if i not in G1_LEG_SLOTS and i not in G1_23_INVALID_SDK_SLOTS]
        assert all(cfg.kp[i] > 0 for i in non_leg_valid)

    def test_freeze_legs_ignored_with_controller(self):
        cfg = _ah_config(freeze_legs=True, controller="GrootLocomotionController")
        non_invalid_legs = [i for i in G1_LEG_SLOTS if i not in G1_23_INVALID_SDK_SLOTS]
        assert all(cfg.kp[i] > 0 for i in non_invalid_legs)

    def test_gain_mutation_isolated_between_instances(self):
        cfg1 = _ah_config()
        cfg2 = _ah_config()
        cfg1.kp[0] = 999.0
        assert cfg2.kp[0] != 999.0


class TestEmbodimentFeatures:
    def test_amazing_hand_per_motor_features(self):
        with _robot(**AH_EMBODIMENT, hand_representation="per_motor") as robot:
            hand_keys = HAND_SPECS["amazing_hand"].joint_keys()
            assert list(robot.action_features) == [*BODY_KEYS, *HEAD_KEYS, *hand_keys]
            assert list(robot.observation_features) == list(robot.action_features)
            assert len(robot.action_features) == 47

    def test_amazing_hand_closure_features(self):
        with _robot(**AH_EMBODIMENT) as robot:
            assert list(robot.action_features) == [*BODY_KEYS, *HEAD_KEYS, *HAND_CLOSURE_KEYS]
            assert len(robot.observation_features) == 33

    def test_amazing_hand_closure_controller_mode_features(self):
        with _robot(controller="GrootLocomotionController", **AH_EMBODIMENT) as robot:
            state = [*ARM_KEYS, *HEAD_KEYS, *HAND_CLOSURE_KEYS]
            assert list(robot.observation_features) == state
            assert list(robot.action_features) == [*state, *NAV_KEYS, BASE_HEIGHT_KEY]
            assert len(robot.observation_features) == 18
            assert len(robot.action_features) == 22

    def test_amazing_hand_per_motor_controller_mode_features(self):
        with _robot(
            controller="HolosomaLocomotionController", hand_representation="per_motor", **AH_EMBODIMENT
        ) as robot:
            state = [*ARM_KEYS, *HEAD_KEYS, *HAND_SPECS["amazing_hand"].joint_keys()]
            assert list(robot.observation_features) == state
            assert list(robot.action_features) == [*state, *NAV_KEYS]

    @pytest.mark.parametrize("end_effector", ["dex1", "dex3"])
    def test_dex_features(self, end_effector):
        with _robot(end_effector=end_effector, hand_representation="per_motor") as robot:
            assert list(robot.action_features) == [*BODY_KEYS, *HAND_SPECS[end_effector].joint_keys()]
        with _robot(end_effector=end_effector) as robot:
            assert list(robot.action_features) == [*BODY_KEYS, *HAND_CLOSURE_KEYS]

    def test_teleop_keys_cover_every_embodiment(self):
        for embodiment, _ in VALID_EMBODIMENTS:
            for representation in ("closure", "per_motor"):
                with _robot(hand_representation=representation, **embodiment) as robot:
                    assert set(robot.action_features) <= set(TELEOP_ACTION_KEYS)

    def test_robot_name_is_robot_type(self):
        with _robot(**AH_EMBODIMENT) as robot:
            assert robot.name == robot.config.robot_type == "unitree_g1_23dof_amazing_hand_d455_pan_tilt"
            assert robot.headhand is not None
            assert robot.dex_hand is None


class TestEmbodimentDatasetFeatures:
    def test_action_feature_shape_and_names(self):
        with _robot(**AH_EMBODIMENT, hand_representation="per_motor") as robot:
            ds_features = hw_to_dataset_features(robot.action_features, "action")
        assert ds_features["action"]["shape"] == (47,)
        assert ds_features["action"]["names"] == list(robot.action_features)

    def test_build_dataset_frame_with_teleop_keys(self):
        with _robot(**AH_EMBODIMENT, hand_representation="per_motor") as robot:
            ds_features = hw_to_dataset_features(robot.action_features, "action")
        values = dict.fromkeys(TELEOP_ACTION_KEYS, 0.0)
        frame = build_dataset_frame(ds_features, values, "action")
        assert frame["action"].shape == (47,)

    def test_build_closure_dataset_frame_with_teleop_keys(self):
        with _robot(**AH_EMBODIMENT) as robot:
            ds_features = hw_to_dataset_features(robot.action_features, "action")
        values = dict.fromkeys(TELEOP_ACTION_KEYS, 0.0)
        values["kRightHand.closure"] = 0.75
        frame = build_dataset_frame(ds_features, values, "action")
        assert frame["action"].shape == (33,)
        assert frame["action"][ds_features["action"]["names"].index("kRightHand.closure")] == 0.75


class TestMakeRobotFromConfig:
    def test_make_robot_from_config_resolves_unitree_g1(self, tmp_path):
        from lerobot.robots.utils import make_robot_from_config

        with (
            patch("lerobot.robots.unitree_g1.unitree_g1.require_package", return_value=None),
            patch("lerobot.robots.unitree_g1.unitree_g1.make_cameras_from_configs", return_value={}),
        ):
            cfg = _ah_config(calibration_dir=tmp_path, id="test")
            robot = make_robot_from_config(cfg)
            assert type(robot).__name__ == "UnitreeG1"
            assert robot.name == "unitree_g1_23dof_amazing_hand_d455_pan_tilt"
