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

"""Tests for G1AhConfig and the device-class fallback resolution. No hardware/SDK required."""

from unittest.mock import patch

import pytest

from lerobot.robots.g1ah.config_g1ah import G1AhConfig
from lerobot.robots.g1ah.g1ah_joints import (
    ALL_ACTION_KEYS,
    G1_23_INVALID_SDK_SLOTS,
    G1_23_LEG_SLOTS,
    TELEOP_ACTION_KEYS,
)
from lerobot.robots.g1ah.g1ah_zmq import HEADHAND_CMD_PORT, HEADHAND_STATE_PORT
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features


class TestG1AhConfigDefaults:
    def test_defaults(self):
        cfg = G1AhConfig()
        assert cfg.revision == "rev_1_0"
        assert cfg.is_simulation is False
        assert cfg.headhand_state_port == HEADHAND_STATE_PORT
        assert cfg.headhand_cmd_port == HEADHAND_CMD_PORT
        assert HEADHAND_STATE_PORT == 6003
        assert HEADHAND_CMD_PORT == 6002

    def test_type(self):
        assert G1AhConfig().type == "g1_23dof_ah8_d455_2dof"

    def test_robot_type_name_per_revision(self):
        assert G1AhConfig(revision="base").robot_type_name == "g1_23dof_ah8_d455_2dof"
        assert G1AhConfig(revision="rev_1_0").robot_type_name == "g1_23dof_ah8_d455_2dof_rev_1_0"

    def test_invalid_revision_raises(self):
        with pytest.raises(ValueError):
            G1AhConfig(revision="not_a_revision")


class TestG1AhConfigGains:
    def test_invalid_slots_zeroed(self):
        cfg = G1AhConfig()
        for i in G1_23_INVALID_SDK_SLOTS:
            assert cfg.kp[i] == 0.0
            assert cfg.kd[i] == 0.0

    def test_valid_slots_nonzero(self):
        cfg = G1AhConfig()
        valid_slots = [i for i in range(29) if i not in G1_23_INVALID_SDK_SLOTS]
        assert all(cfg.kp[i] > 0 for i in valid_slots)
        assert all(cfg.kd[i] > 0 for i in valid_slots)

    def test_freeze_legs_zeroes_leg_slots_only(self):
        cfg = G1AhConfig(freeze_legs=True, controller=None)
        for i in G1_23_LEG_SLOTS:
            assert cfg.kp[i] == 0.0
            assert cfg.kd[i] == 0.0
        non_leg_valid = [
            i for i in range(29) if i not in G1_23_LEG_SLOTS and i not in G1_23_INVALID_SDK_SLOTS
        ]
        assert all(cfg.kp[i] > 0 for i in non_leg_valid)

    def test_freeze_legs_ignored_with_controller(self):
        cfg = G1AhConfig(freeze_legs=True, controller="GrootLocomotionController")
        non_invalid_legs = [i for i in G1_23_LEG_SLOTS if i not in G1_23_INVALID_SDK_SLOTS]
        assert all(cfg.kp[i] > 0 for i in non_invalid_legs)

    def test_gain_mutation_isolated_between_instances(self):
        cfg1 = G1AhConfig()
        cfg2 = G1AhConfig()
        cfg1.kp[0] = 999.0
        assert cfg2.kp[0] != 999.0


class TestG1AhConfigValidation:
    def test_bad_kp_length_raises(self):
        with pytest.raises(ValueError):
            G1AhConfig(kp=[1.0] * 28)

    def test_bad_head_default_positions_length_raises(self):
        with pytest.raises(ValueError):
            G1AhConfig(head_default_positions=[0.0])

    def test_bad_hand_default_positions_length_raises(self):
        with pytest.raises(ValueError):
            G1AhConfig(hand_default_positions=[0.0] * 15)


class TestG1AhDatasetFeatures:
    def test_action_feature_shape_and_names(self):
        ds_features = hw_to_dataset_features(dict.fromkeys(ALL_ACTION_KEYS, float), "action")
        assert ds_features["action"]["shape"] == (41,)
        assert ds_features["action"]["names"] == list(ALL_ACTION_KEYS)

    def test_build_dataset_frame_with_teleop_keys(self):
        ds_features = hw_to_dataset_features(dict.fromkeys(ALL_ACTION_KEYS, float), "action")
        values = dict.fromkeys(TELEOP_ACTION_KEYS, 0.0)
        frame = build_dataset_frame(ds_features, values, "action")
        assert frame["action"].shape == (41,)


class TestG1AhDeviceClassFallback:
    def test_make_robot_from_config_resolves_g1ah(self, tmp_path):
        from lerobot.robots.utils import make_robot_from_config

        with (
            patch("lerobot.robots.unitree_g1.unitree_g1.require_package", return_value=None),
            patch("lerobot.robots.unitree_g1.unitree_g1.make_cameras_from_configs", return_value={}),
        ):
            cfg = G1AhConfig(calibration_dir=tmp_path, id="test")
            robot = make_robot_from_config(cfg)
            assert type(robot).__name__ == "G1Ah"
