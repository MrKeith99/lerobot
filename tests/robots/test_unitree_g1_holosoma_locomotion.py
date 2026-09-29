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

"""Tests for the Holosoma locomotion controller's navigation command. The ONNX policy is stubbed."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from lerobot.robots.unitree_g1.g1_utils import NAV_KEYS, default_remote_input  # noqa: E402
from lerobot.robots.unitree_g1.holosoma_locomotion import HolosomaLocomotionController  # noqa: E402


def _stub_policy():
    policy = MagicMock()
    policy.get_inputs.return_value = [MagicMock(name="obs")]
    policy.run.return_value = [np.zeros((1, 29), dtype=np.float32)]
    return policy


@pytest.fixture
def controller():
    gains = np.ones(29, dtype=np.float32)
    with patch(
        "lerobot.robots.unitree_g1.holosoma_locomotion.load_policy",
        return_value=(_stub_policy(), gains, gains),
    ):
        yield HolosomaLocomotionController()


def _lowstate():
    state = MagicMock()
    state.motor_state = [MagicMock(q=0.0, dq=0.0) for _ in range(35)]
    state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    state.imu_state.gyroscope = [0.0, 0.0, 0.0]
    return state


def test_nav_command_is_used_and_clipped(controller):
    action = {**default_remote_input(), **dict(zip(NAV_KEYS, (0.8, -0.2, 0.5), strict=True))}
    controller.run_step(action, _lowstate())
    np.testing.assert_allclose(controller.cmd, [0.3, -0.2, 0.5])
    np.testing.assert_allclose(controller.obs[32:35], [0.5, 0.3, -0.2])


def test_nav_command_dead_zone(controller):
    action = {**default_remote_input(), **dict(zip(NAV_KEYS, (0.05, 0.2, -0.05), strict=True))}
    controller.run_step(action, _lowstate())
    np.testing.assert_allclose(controller.cmd, [0.0, 0.2, 0.0])


def test_nav_command_matches_equivalent_remote_axes(controller):
    remote = {**default_remote_input(), "remote.lx": -0.2, "remote.ly": 0.25, "remote.rx": -0.6}
    controller.run_step(remote, _lowstate())
    from_remote = controller.cmd.copy()
    nav = {**default_remote_input(), **dict(zip(NAV_KEYS, (0.25, 0.2, 0.6), strict=True))}
    controller.run_step(nav, _lowstate())
    np.testing.assert_allclose(controller.cmd, from_remote)


def test_nav_command_takes_precedence_over_remote_axes(controller):
    action = {**default_remote_input(), "remote.ly": 1.0, **dict.fromkeys(NAV_KEYS, 0.0)}
    controller.run_step(action, _lowstate())
    np.testing.assert_allclose(controller.cmd, [0.0, 0.0, 0.0])
