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

"""Tests for the GR00T locomotion controller's base-height command. The ONNX policies are stubbed."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from lerobot.robots.unitree_g1.g1_utils import (  # noqa: E402
    BASE_HEIGHT_KEY,
    GROOT_BASE_HEIGHT_DEFAULT,
    GROOT_BASE_HEIGHT_RANGE,
    GROOT_BASE_HEIGHT_RATE,
    NAV_KEYS,
    default_remote_input,
    nav_from_remote,
)
from lerobot.robots.unitree_g1.gr00t_locomotion import CONTROL_DT, GrootLocomotionController  # noqa: E402


def _stub_policy():
    policy = MagicMock()
    policy.get_inputs.return_value = [MagicMock(name="obs")]
    policy.run.return_value = [np.zeros((1, 15), dtype=np.float32)]
    return policy


@pytest.fixture
def controller():
    with patch(
        "lerobot.robots.unitree_g1.gr00t_locomotion.load_groot_policies",
        return_value=(_stub_policy(), _stub_policy()),
    ):
        yield GrootLocomotionController()


def _lowstate():
    state = MagicMock()
    state.motor_state = [MagicMock(q=0.0, dq=0.0) for _ in range(35)]
    state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    state.imu_state.gyroscope = [0.0, 0.0, 0.0]
    return state


def test_default_height(controller):
    controller.run_step(default_remote_input(), _lowstate())
    assert controller.groot_height_cmd == pytest.approx(GROOT_BASE_HEIGHT_DEFAULT)


def test_explicit_height_command_is_used_and_observed(controller):
    action = {**default_remote_input(), BASE_HEIGHT_KEY: 0.62}
    controller.run_step(action, _lowstate())
    assert controller.groot_height_cmd == pytest.approx(0.62)
    assert controller.groot_obs_single[3] == pytest.approx(0.62)


@pytest.mark.parametrize(
    "value, expected", [(0.1, GROOT_BASE_HEIGHT_RANGE[0]), (1.5, GROOT_BASE_HEIGHT_RANGE[1])]
)
def test_explicit_height_command_is_clipped(controller, value, expected):
    controller.run_step({**default_remote_input(), BASE_HEIGHT_KEY: value}, _lowstate())
    assert controller.groot_height_cmd == pytest.approx(expected)


def test_explicit_height_overrides_buttons(controller):
    action = {**default_remote_input(), "remote.button.0": 1.0, BASE_HEIGHT_KEY: 0.6}
    for _ in range(10):
        controller.run_step(action, _lowstate())
    assert controller.groot_height_cmd == pytest.approx(0.6)


@pytest.mark.parametrize("button, sign", [("remote.button.0", 1.0), ("remote.button.4", -1.0)])
def test_buttons_still_integrate_without_explicit_height(controller, button, sign):
    action = {**default_remote_input(), button: 1.0}
    for _ in range(10):
        controller.run_step(action, _lowstate())
    expected = GROOT_BASE_HEIGHT_DEFAULT + sign * 10 * GROOT_BASE_HEIGHT_RATE * CONTROL_DT
    assert controller.groot_height_cmd == pytest.approx(expected)


def test_reset_restores_default_height(controller):
    controller.run_step({**default_remote_input(), BASE_HEIGHT_KEY: 0.55}, _lowstate())
    controller.reset()
    assert controller.groot_height_cmd == pytest.approx(GROOT_BASE_HEIGHT_DEFAULT)


def test_nav_command_sets_velocity_command(controller):
    action = {**default_remote_input(), **dict(zip(NAV_KEYS, (0.4, -0.1, 0.3), strict=True))}
    controller.run_step(action, _lowstate())
    np.testing.assert_allclose(controller.cmd, [0.4, -0.1, 0.3], rtol=1e-6)


def test_nav_command_matches_remote_axes_mapping(controller):
    remote = {**default_remote_input(), "remote.lx": 0.3, "remote.ly": 0.7, "remote.rx": -0.5}
    controller.run_step(remote, _lowstate())
    from_remote = controller.cmd.copy()
    nav = {**default_remote_input(), **nav_from_remote(0.3, 0.7, -0.5)}
    controller.run_step(nav, _lowstate())
    np.testing.assert_allclose(controller.cmd, from_remote)
    np.testing.assert_allclose(from_remote, [0.7, -0.3, 0.5], rtol=1e-6)


def test_nav_command_takes_precedence_over_remote_axes(controller):
    action = {**default_remote_input(), "remote.ly": 1.0, **dict.fromkeys(NAV_KEYS, 0.0)}
    controller.run_step(action, _lowstate())
    np.testing.assert_allclose(controller.cmd, [0.0, 0.0, 0.0])
