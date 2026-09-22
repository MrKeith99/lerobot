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
from lerobot.teleoperators.unitree_g1_ah_gamepad import (
    UnitreeG1AhGamepadTeleop,
    UnitreeG1AhGamepadTeleopConfig,
)
from lerobot.teleoperators.utils import TeleopEvents, make_teleoperator_from_config
from lerobot.utils.errors import DeviceNotConnectedError

_MODULE = "lerobot.teleoperators.unitree_g1_ah_gamepad.unitree_g1_ah_gamepad"


class FakeInput:
    def __init__(self, layout, deadzone):
        self.layout = layout
        self.deadzone = deadzone
        self.axes: dict[int, float] = {}
        self.buttons: set[int] = set()
        self.hat_value: tuple[int, int] = (0, 0)
        self.episode_end_status = None
        self.intervention_flag = False
        self.running = True
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def update(self) -> None:
        pass

    def axis(self, index: int) -> float:
        return self.axes.get(index, 0.0)

    def button(self, index: int) -> bool:
        return index in self.buttons

    def hat(self) -> tuple[int, int]:
        return self.hat_value

    def should_intervene(self) -> bool:
        return self.intervention_flag

    def get_episode_end_status(self):
        status = self.episode_end_status
        self.episode_end_status = None
        return status

    @property
    def is_running(self) -> bool:
        return self.running


@pytest.fixture
def teleop():
    with patch(f"{_MODULE}.UnitreeG1AhGamepadInput", FakeInput):
        t = UnitreeG1AhGamepadTeleop(UnitreeG1AhGamepadTeleopConfig())
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


def test_hat_up_tilts_and_saturates(teleop):
    teleop.gamepad.hat_value = (0, 1)
    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        first_tilt = None
        for _ in range(200):
            clock["t"] += 0.05
            action = teleop.get_action()
            if first_tilt is None:
                first_tilt = action["d455_joint.q"]
        assert first_tilt is not None
        assert first_tilt > 0.0
        assert action["d455_joint.q"] == pytest.approx(0.8, abs=1e-6)


def test_hat_pan_saturates_both_directions(teleop):
    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.hat_value = (1, 0)
        for _ in range(200):
            clock["t"] += 0.05
            action = teleop.get_action()
        assert action["xl330_joint.q"] == pytest.approx(-0.7, abs=1e-6)

        teleop.gamepad.hat_value = (-1, 0)
        for _ in range(400):
            clock["t"] += 0.05
            action = teleop.get_action()
        assert action["xl330_joint.q"] == pytest.approx(0.7, abs=1e-6)


def test_rb_closes_right_hand_only(teleop):
    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.buttons.add(teleop.config.layout.button_rb)
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


def test_rb_release_returns_to_open(teleop):
    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.buttons.add(teleop.config.layout.button_rb)
        for _ in range(50):
            clock["t"] += 0.02
            teleop.get_action()
        teleop.gamepad.buttons.discard(teleop.config.layout.button_rb)
        for _ in range(50):
            clock["t"] += 0.02
            action = teleop.get_action()

        open_right = hand_pose_rad("right", False)
        for name, expected in zip(
            (f"right_hand_finger{i}_motor{j}" for i in range(1, 5) for j in range(1, 3)),
            open_right,
            strict=True,
        ):
            assert action[f"{name}.q"] == pytest.approx(expected, abs=1e-3)


def test_body_keys_never_change(teleop):
    default = default_action()
    body_keys = [k for k in default if k.endswith(".q") and not k.startswith(("xl330", "d455"))]
    body_keys = [k for k in body_keys if "hand" not in k]

    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.hat_value = (1, 1)
        teleop.gamepad.buttons.add(teleop.config.layout.button_rb)
        teleop.gamepad.buttons.add(teleop.config.layout.button_lb)
        for _ in range(20):
            clock["t"] += 0.05
            action = teleop.get_action()

    for key in body_keys:
        assert action[key] == pytest.approx(default[key])


def test_remote_axes_passthrough_with_sign_convention(teleop):
    layout = teleop.config.layout
    teleop.gamepad.axes[layout.left_x] = 0.5
    teleop.gamepad.axes[layout.left_y] = 0.5
    teleop.gamepad.axes[layout.right_x] = -0.3
    teleop.gamepad.axes[layout.right_y] = -0.3

    action = teleop.get_action()
    assert action["remote.lx"] == pytest.approx(0.5)
    assert action["remote.ly"] == pytest.approx(-0.5)
    assert action["remote.rx"] == pytest.approx(-0.3)
    assert action["remote.ry"] == pytest.approx(0.3)


def test_emit_remote_axes_false_zeros_out():
    with patch(f"{_MODULE}.UnitreeG1AhGamepadInput", FakeInput):
        t = UnitreeG1AhGamepadTeleop(UnitreeG1AhGamepadTeleopConfig(emit_remote_axes=False))
        t.connect()
        t.gamepad.axes[t.config.layout.left_x] = 0.9
        action = t.get_action()
        for key in REMOTE_AXES:
            assert key in action
            assert action[key] == pytest.approx(0.0)
        t.disconnect()


def test_initial_positions_override_applied():
    cfg = UnitreeG1AhGamepadTeleopConfig(initial_positions={"xl330_joint.q": 0.3})
    with patch(f"{_MODULE}.UnitreeG1AhGamepadInput", FakeInput):
        t = UnitreeG1AhGamepadTeleop(cfg)
        t.connect()
        action = t.get_action()
        assert action["xl330_joint.q"] == pytest.approx(0.3)
        t.disconnect()


def test_initial_positions_invalid_key_raises():
    cfg = UnitreeG1AhGamepadTeleopConfig(initial_positions={"not_a_real_key.q": 0.0})
    with pytest.raises(ValueError):
        UnitreeG1AhGamepadTeleop(cfg)


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


def test_make_teleoperator_from_config_returns_class_without_connecting():
    teleop = make_teleoperator_from_config(UnitreeG1AhGamepadTeleopConfig())
    assert isinstance(teleop, UnitreeG1AhGamepadTeleop)
    assert not teleop.is_connected


def test_disconnect_idempotent(teleop):
    teleop.disconnect()
    teleop.disconnect()
    assert not teleop.is_connected


def test_get_action_before_connect_raises():
    teleop = UnitreeG1AhGamepadTeleop(UnitreeG1AhGamepadTeleopConfig())
    with pytest.raises(DeviceNotConnectedError):
        teleop.get_action()
