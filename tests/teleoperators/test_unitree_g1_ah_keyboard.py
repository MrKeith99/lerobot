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

from __future__ import annotations

from unittest.mock import patch

import pytest

from lerobot.robots.unitree_g1.g1_utils import REMOTE_AXES
from lerobot.robots.unitree_g1_ah.g1_ah_joints import TELEOP_ACTION_KEYS, default_action, hand_pose_rad
from lerobot.teleoperators.unitree_g1_ah_keyboard import (
    UnitreeG1AhKeyboardTeleop,
    UnitreeG1AhKeyboardTeleopConfig,
)
from lerobot.teleoperators.unitree_g1_ah_keyboard.keyboard_input import UnitreeG1AhKeyboardInput
from lerobot.teleoperators.utils import TeleopEvents, make_teleoperator_from_config
from lerobot.utils.errors import DeviceNotConnectedError

_MODULE = "lerobot.teleoperators.unitree_g1_ah_keyboard.unitree_g1_ah_keyboard"
# get_action/time.perf_counter are inherited unmodified from UnitreeG1AhGamepadTeleop, so
# `time` must be patched in its defining module, not in the (time-import-less) keyboard one.
_PARENT_MODULE = "lerobot.teleoperators.unitree_g1_ah_gamepad.unitree_g1_ah_gamepad"


class FakeInput(UnitreeG1AhKeyboardInput):
    """`UnitreeG1AhKeyboardInput` with `start`/`stop` stubbed out (no real listener)."""

    def __init__(self, config):
        super().__init__(config)
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


# ── UnitreeG1AhKeyboardInput unit tests (no pynput, driven via press/release) ──


@pytest.fixture
def keyboard_input():
    return UnitreeG1AhKeyboardInput(UnitreeG1AhKeyboardTeleopConfig())


def test_hat_arrow_mapping(keyboard_input):
    assert keyboard_input.hat() == (0, 0)
    keyboard_input.press("left")
    assert keyboard_input.hat() == (-1, 0)
    keyboard_input.release("left")
    keyboard_input.press("right")
    assert keyboard_input.hat() == (1, 0)
    keyboard_input.release("right")
    keyboard_input.press("up")
    assert keyboard_input.hat() == (0, 1)
    keyboard_input.release("up")
    keyboard_input.press("down")
    assert keyboard_input.hat() == (0, -1)


def test_button_lb_rb_via_q_e(keyboard_input):
    layout = keyboard_input.config.layout
    assert not keyboard_input.button(layout.button_lb)
    assert not keyboard_input.button(layout.button_rb)

    keyboard_input.press("q")
    assert keyboard_input.button(layout.button_lb)
    assert not keyboard_input.button(layout.button_rb)

    keyboard_input.press("e")
    assert keyboard_input.button(layout.button_rb)

    keyboard_input.release("q")
    assert not keyboard_input.button(layout.button_lb)
    assert keyboard_input.button(layout.button_rb)


def test_axis_sign_convention_left_stick(keyboard_input):
    layout = keyboard_input.config.layout
    value = keyboard_input.config.remote_axis_value

    keyboard_input.press("w")
    assert keyboard_input.axis(layout.left_y) == pytest.approx(-value)
    keyboard_input.release("w")
    keyboard_input.press("s")
    assert keyboard_input.axis(layout.left_y) == pytest.approx(value)
    keyboard_input.release("s")

    keyboard_input.press("a")
    assert keyboard_input.axis(layout.left_x) == pytest.approx(-value)
    keyboard_input.release("a")
    keyboard_input.press("d")
    assert keyboard_input.axis(layout.left_x) == pytest.approx(value)


def test_axis_sign_convention_right_stick(keyboard_input):
    layout = keyboard_input.config.layout
    value = keyboard_input.config.remote_axis_value

    keyboard_input.press("i")
    assert keyboard_input.axis(layout.right_y) == pytest.approx(-value)
    keyboard_input.release("i")
    keyboard_input.press("k")
    assert keyboard_input.axis(layout.right_y) == pytest.approx(value)
    keyboard_input.release("k")

    keyboard_input.press("j")
    assert keyboard_input.axis(layout.right_x) == pytest.approx(-value)
    keyboard_input.release("j")
    keyboard_input.press("l")
    assert keyboard_input.axis(layout.right_x) == pytest.approx(value)


def test_episode_events_latched_until_release(keyboard_input):
    assert keyboard_input.get_episode_end_status() is None
    keyboard_input.press("y")
    assert keyboard_input.get_episode_end_status() == TeleopEvents.SUCCESS
    assert keyboard_input.get_episode_end_status() == TeleopEvents.SUCCESS
    keyboard_input.release("y")
    assert keyboard_input.get_episode_end_status() is None

    keyboard_input.press("n")
    assert keyboard_input.get_episode_end_status() == TeleopEvents.FAILURE
    keyboard_input.release("n")

    keyboard_input.press("r")
    assert keyboard_input.get_episode_end_status() == TeleopEvents.RERECORD_EPISODE
    keyboard_input.release("r")
    assert keyboard_input.get_episode_end_status() is None


def test_space_held_is_intervention(keyboard_input):
    assert not keyboard_input.should_intervene()
    keyboard_input.press("space")
    assert keyboard_input.should_intervene()
    keyboard_input.release("space")
    assert not keyboard_input.should_intervene()


def test_esc_stops_running(keyboard_input):
    assert keyboard_input.is_running
    keyboard_input.press("esc")
    keyboard_input.release("esc")
    assert not keyboard_input.is_running


def test_start_without_pynput_logs_warning_and_runs_with_no_keys(caplog):
    ki = UnitreeG1AhKeyboardInput(UnitreeG1AhKeyboardTeleopConfig())
    with (
        patch(
            "lerobot.teleoperators.unitree_g1_ah_keyboard.keyboard_input.pynput_can_capture",
            return_value=False,
        ),
        caplog.at_level("WARNING"),
    ):
        ki.start()
    assert "unavailable" in caplog.text
    assert ki.hat() == (0, 0)
    assert ki.is_running


# ── Teleop-level tests (FakeInput-patched, mirroring the gamepad test suite) ──


@pytest.fixture
def teleop():
    with patch(f"{_MODULE}.UnitreeG1AhKeyboardInput", FakeInput):
        t = UnitreeG1AhKeyboardTeleop(UnitreeG1AhKeyboardTeleopConfig())
        t.connect()
        yield t
        if t.is_connected:
            t.disconnect()


def test_action_features_are_teleop_action_keys(teleop):
    assert set(teleop.action_features) == set(TELEOP_ACTION_KEYS)
    assert len(teleop.action_features) == 45


def test_get_action_keys_match_action_features(teleop):
    action = teleop.get_action()
    assert set(action) == set(teleop.action_features) == set(TELEOP_ACTION_KEYS)


def test_idle_action_matches_default_and_zero_remote(teleop):
    action = teleop.get_action()
    default = default_action()
    for key, value in default.items():
        assert action[key] == pytest.approx(value)
    for key in REMOTE_AXES:
        assert action[key] == pytest.approx(0.0)


def test_arrow_keys_move_head_same_sign_as_gamepad(teleop):
    clock = {"t": 0.0}
    with patch(f"{_PARENT_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.press("up")
        for _ in range(200):
            clock["t"] += 0.05
            action = teleop.get_action()
        assert action["d455_joint.q"] == pytest.approx(0.8, abs=1e-6)
        teleop.gamepad.release("up")

        teleop.gamepad.press("left")
        for _ in range(400):
            clock["t"] += 0.05
            action = teleop.get_action()
        assert action["xl330_joint.q"] == pytest.approx(0.7, abs=1e-6)


def test_e_held_closes_right_hand(teleop):
    clock = {"t": 0.0}
    with patch(f"{_PARENT_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.press("e")
        duration_s = 1.0 / teleop.config.hand_blend_per_s
        steps = 50
        dt = duration_s / steps
        for _ in range(steps):
            clock["t"] += dt
            action = teleop.get_action()

        closed_right = hand_pose_rad("right", True)
        for name, expected in zip(
            (f"right_hand_finger{i}_motor{j}" for i in range(1, 5) for j in range(1, 3)),
            closed_right,
            strict=True,
        ):
            assert action[f"{name}.q"] == pytest.approx(expected, abs=1e-3)

        default = default_action()
        for i in range(1, 5):
            for j in range(1, 3):
                key = f"left_hand_finger{i}_motor{j}.q"
                assert action[key] == pytest.approx(default[key])


def test_w_gives_positive_remote_ly(teleop):
    teleop.gamepad.press("w")
    action = teleop.get_action()
    assert action["remote.ly"] > 0.0


def test_emit_remote_axes_false_zeros_out():
    with patch(f"{_MODULE}.UnitreeG1AhKeyboardInput", FakeInput):
        t = UnitreeG1AhKeyboardTeleop(UnitreeG1AhKeyboardTeleopConfig(emit_remote_axes=False))
        t.connect()
        t.gamepad.press("d")
        action = t.get_action()
        for key in REMOTE_AXES:
            assert key in action
            assert action[key] == pytest.approx(0.0)
        t.disconnect()


def test_get_teleop_events_maps_success_failure_rerecord(teleop):
    teleop.gamepad.episode_end_status = TeleopEvents.SUCCESS
    events = teleop.get_teleop_events()
    assert events[TeleopEvents.SUCCESS] is True
    assert events[TeleopEvents.RERECORD_EPISODE] is False
    assert events[TeleopEvents.TERMINATE_EPISODE] is False

    teleop.gamepad.episode_end_status = TeleopEvents.FAILURE
    events = teleop.get_teleop_events()
    assert events[TeleopEvents.SUCCESS] is False
    assert events[TeleopEvents.TERMINATE_EPISODE] is True

    teleop.gamepad.episode_end_status = TeleopEvents.RERECORD_EPISODE
    events = teleop.get_teleop_events()
    assert events[TeleopEvents.RERECORD_EPISODE] is True
    assert events[TeleopEvents.TERMINATE_EPISODE] is True


def test_disconnect_idempotent(teleop):
    teleop.disconnect()
    teleop.disconnect()
    assert not teleop.is_connected


def test_get_action_before_connect_raises():
    teleop = UnitreeG1AhKeyboardTeleop(UnitreeG1AhKeyboardTeleopConfig())
    with pytest.raises(DeviceNotConnectedError):
        teleop.get_action()


def test_make_teleoperator_from_config_returns_class_without_importing_pynput():
    import sys

    sys.modules.pop("pynput", None)
    teleop = make_teleoperator_from_config(UnitreeG1AhKeyboardTeleopConfig())
    assert isinstance(teleop, UnitreeG1AhKeyboardTeleop)
    assert not teleop.is_connected
    assert "pynput" not in sys.modules
