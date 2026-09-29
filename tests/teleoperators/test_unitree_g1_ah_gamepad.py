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

from lerobot.robots.unitree_g1.g1_utils import REMOTE_AXES, REMOTE_BUTTONS, REMOTE_KEYS
from lerobot.robots.unitree_g1_ah.g1_ah_joints import (
    BODY_KEYS,
    HAND_CLOSURE_KEYS,
    HEAD_KEYS,
    TELEOP_ACTION_KEYS,
    closure_to_hand_q,
    default_action,
    hand_motor_names,
    hand_pose_rad,
)
from lerobot.teleoperators.unitree_g1_ah_gamepad import (
    UnitreeG1AhGamepadTeleop,
    UnitreeG1AhGamepadTeleopConfig,
)
from lerobot.teleoperators.unitree_g1_ah_gamepad.config_unitree_g1_ah_gamepad import GamepadLayout
from lerobot.teleoperators.unitree_g1_ah_gamepad.gamepad_input import _dpad_from_buttons
from lerobot.teleoperators.utils import TeleopEvents, make_teleoperator_from_config
from lerobot.utils.errors import DeviceNotConnectedError
from tests.utils import skip_if_package_missing

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
        if self.layout.dpad_up is not None:
            return _dpad_from_buttons(self, self.layout)
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
    assert len(teleop.action_features) == 70


def test_get_action_keys_match_action_features(teleop):
    action = teleop.get_action()
    assert set(action) == set(teleop.action_features) == set(TELEOP_ACTION_KEYS)


def test_idle_action_matches_default_and_zero_remote(teleop):
    action = teleop.get_action()
    default = default_action()
    for key, value in default.items():
        assert action[key] == pytest.approx(value)
    for key in REMOTE_KEYS:
        assert action[key] == pytest.approx(0.0)
    for key in HAND_CLOSURE_KEYS:
        assert action[key] == 0.0


def test_triggers_drive_waist_buttons_not_hands(teleop):
    layout = teleop.config.layout
    default = default_action()

    teleop.gamepad.axes[layout.trigger_left] = 1.0
    action = teleop.get_action()
    assert action["remote.button.0"] == 1.0
    assert action["remote.button.4"] == 0.0
    for side in ("left", "right"):
        for name in hand_motor_names(side):
            assert action[f"{name}.q"] == pytest.approx(default[f"{name}.q"])

    teleop.gamepad.axes[layout.trigger_left] = -1.0
    teleop.gamepad.axes[layout.trigger_right] = 1.0
    action = teleop.get_action()
    assert action["remote.button.0"] == 0.0
    assert action["remote.button.4"] == 1.0

    teleop.gamepad.axes[layout.trigger_right] = -1.0
    teleop.gamepad.buttons.add(layout.button_rb)
    teleop.gamepad.buttons.add(layout.button_lb)
    action = teleop.get_action()
    for key in REMOTE_BUTTONS:
        assert action[key] == 0.0


def test_hat_up_tilts_and_saturates(teleop):
    layout = teleop.config.layout
    teleop.gamepad.buttons.add(layout.dpad_up)
    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        first_tilt = None
        for _ in range(200):
            clock["t"] += 0.05
            action = teleop.get_action()
            if first_tilt is None:
                first_tilt = action["kHeadPitch.q"]
        assert first_tilt is not None
        assert first_tilt > 0.0
        assert action["kHeadPitch.q"] == pytest.approx(0.8, abs=1e-6)


def test_hat_pan_saturates_both_directions(teleop):
    layout = teleop.config.layout
    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.buttons.add(layout.dpad_right)
        for _ in range(200):
            clock["t"] += 0.05
            action = teleop.get_action()
        assert action["kHeadYaw.q"] == pytest.approx(-0.7, abs=1e-6)

        teleop.gamepad.buttons.discard(layout.dpad_right)
        teleop.gamepad.buttons.add(layout.dpad_left)
        for _ in range(400):
            clock["t"] += 0.05
            action = teleop.get_action()
        assert action["kHeadYaw.q"] == pytest.approx(0.7, abs=1e-6)


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
            hand_motor_names("right"),
            closed_right,
            strict=True,
        ):
            assert action[f"{name}.q"] == pytest.approx(expected, abs=1e-3)

        default = default_action()
        for name in hand_motor_names("left"):
            key = f"{name}.q"
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
            hand_motor_names("right"),
            open_right,
            strict=True,
        ):
            assert action[f"{name}.q"] == pytest.approx(expected, abs=1e-3)


def _hold(teleop, clock, button, seconds, steps=50):
    if button is not None:
        teleop.gamepad.buttons.add(button)
    for _ in range(steps):
        clock["t"] += seconds / steps
        action = teleop.get_action()
    return action


def test_rb_ramps_right_closure_at_blend_rate(teleop):
    clock = {"t": 0.0}
    rate = teleop.config.hand_blend_per_s
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        half = _hold(teleop, clock, teleop.config.layout.button_rb, 0.5 / rate)
        assert half["kRightHand.closure"] == pytest.approx(0.5, abs=1e-6)
        assert half["kLeftHand.closure"] == 0.0
        full = _hold(teleop, clock, None, 1.0 / rate)
        assert full["kRightHand.closure"] == pytest.approx(1.0)
        assert full["kLeftHand.closure"] == 0.0


@pytest.mark.parametrize("side", ["left", "right"])
def test_closure_matches_per_motor_targets(teleop, side):
    button = teleop.config.layout.button_lb if side == "left" else teleop.config.layout.button_rb
    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.buttons.add(button)
        for _ in range(40):
            clock["t"] += 0.01
            action = teleop.get_action()
            closure = action[f"k{side.capitalize()}Hand.closure"]
            assert 0.0 <= closure <= 1.0
            motors = [action[f"{name}.q"] for name in hand_motor_names(side)]
            assert motors == pytest.approx(closure_to_hand_q(side, closure), abs=1e-9)


def test_lb_release_returns_left_closure_to_zero(teleop):
    clock = {"t": 0.0}
    rate = teleop.config.hand_blend_per_s
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        closed = _hold(teleop, clock, teleop.config.layout.button_lb, 1.0 / rate)
        assert closed["kLeftHand.closure"] == pytest.approx(1.0)
        teleop.gamepad.buttons.discard(teleop.config.layout.button_lb)
        released = _hold(teleop, clock, None, 1.0 / rate)
        assert released["kLeftHand.closure"] == pytest.approx(0.0)


def test_initial_closure_seeds_hand_blend():
    with patch(f"{_MODULE}.UnitreeG1AhGamepadInput", FakeInput):
        t = UnitreeG1AhGamepadTeleop(
            UnitreeG1AhGamepadTeleopConfig(
                initial_positions={"kLeftHand.closure": 1.0, "kRightHand.closure": 0.25}
            )
        )
        t.connect()
        action = t.get_action()
        t.disconnect()
    assert action["kLeftHand.closure"] == pytest.approx(1.0)
    assert action["kRightHand.closure"] == pytest.approx(0.25)
    assert [action[f"{name}.q"] for name in hand_motor_names("left")] == pytest.approx(
        hand_pose_rad("left", True)
    )


def _measured_obs(hand_closure=0.3):
    obs = {key: 0.05 * (i + 1) for i, key in enumerate(BODY_KEYS)}
    obs.update({"kHeadYaw.q": 0.2, "kHeadPitch.q": -0.3})
    for side in ("left", "right"):
        obs.update(
            zip(
                (f"{name}.q" for name in hand_motor_names(side)),
                closure_to_hand_q(side, hand_closure),
                strict=True,
            )
        )
    return obs


def test_first_feedback_sets_targets_to_measured_pose(teleop):
    obs = _measured_obs(hand_closure=0.3)
    teleop.send_feedback(obs)
    action = teleop.get_action()
    for key in (*BODY_KEYS, *HEAD_KEYS):
        assert action[key] == pytest.approx(obs[key])
    for side in ("left", "right"):
        assert action[f"k{side.capitalize()}Hand.closure"] == pytest.approx(0.3)
        assert [action[f"{name}.q"] for name in hand_motor_names(side)] == pytest.approx(
            closure_to_hand_q(side, 0.3)
        )


def test_feedback_closure_keys_take_precedence_over_motor_angles(teleop):
    obs = _measured_obs(hand_closure=0.3)
    obs["kRightHand.closure"] = 0.8
    teleop.send_feedback(obs)
    assert teleop.get_action()["kRightHand.closure"] == pytest.approx(0.8)


def test_feedback_is_latched_once(teleop):
    first = _measured_obs()
    teleop.send_feedback(first)
    teleop.send_feedback(dict.fromkeys(first, -1.0))
    action = teleop.get_action()
    for key in BODY_KEYS:
        assert action[key] == pytest.approx(first[key])


def test_feedback_without_full_body_is_ignored(teleop):
    teleop.send_feedback({"kLeftElbow.q": 1.2})
    assert teleop.get_action()["kLeftElbow.q"] == 0.0
    teleop.send_feedback(_measured_obs())
    assert teleop.get_action()["kLeftElbow.q"] == pytest.approx(_measured_obs()["kLeftElbow.q"])


def test_initial_positions_win_over_feedback():
    with patch(f"{_MODULE}.UnitreeG1AhGamepadInput", FakeInput):
        t = UnitreeG1AhGamepadTeleop(
            UnitreeG1AhGamepadTeleopConfig(initial_positions={"kLeftElbow.q": 0.5, "kLeftHand.closure": 1.0})
        )
        t.connect()
        t.send_feedback(_measured_obs(hand_closure=0.3))
        action = t.get_action()
        t.disconnect()
    assert action["kLeftElbow.q"] == 0.5
    assert action["kLeftHand.closure"] == pytest.approx(1.0)
    assert action["kRightHand.closure"] == pytest.approx(0.3)
    assert action["kRightElbow.q"] == pytest.approx(_measured_obs()["kRightElbow.q"])


def test_dpad_moves_head_from_measured_pose(teleop):
    obs = _measured_obs()
    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.send_feedback(obs)
        teleop.get_action()
        teleop.gamepad.buttons.add(teleop.config.layout.dpad_up)
        clock["t"] += 0.1
        action = teleop.get_action()
    assert action["kHeadPitch.q"] == pytest.approx(obs["kHeadPitch.q"] + teleop.config.head_speed_rad_s * 0.1)
    assert action["kHeadYaw.q"] == pytest.approx(obs["kHeadYaw.q"])


@pytest.mark.parametrize("trigger, sign", [("trigger_left", 1.0), ("trigger_right", -1.0)])
def test_triggers_move_base_height(teleop, trigger, sign):
    from lerobot.robots.unitree_g1.g1_utils import GROOT_BASE_HEIGHT_DEFAULT, GROOT_BASE_HEIGHT_RATE

    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        assert teleop.get_action()["kBaseHeight.cmd"] == pytest.approx(GROOT_BASE_HEIGHT_DEFAULT)
        teleop.gamepad.axes[getattr(teleop.config.layout, trigger)] = 1.0
        clock["t"] += 0.1
        action = teleop.get_action()
    assert action["kBaseHeight.cmd"] == pytest.approx(
        GROOT_BASE_HEIGHT_DEFAULT + sign * 0.1 * GROOT_BASE_HEIGHT_RATE
    )


def test_base_height_clipped_and_seeded(teleop):
    from lerobot.robots.unitree_g1.g1_utils import GROOT_BASE_HEIGHT_RANGE

    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.axes[teleop.config.layout.trigger_left] = 1.0
        for _ in range(100):
            clock["t"] += 0.1
            action = teleop.get_action()
    assert action["kBaseHeight.cmd"] == pytest.approx(GROOT_BASE_HEIGHT_RANGE[1])
    with patch(f"{_MODULE}.UnitreeG1AhGamepadInput", FakeInput):
        seeded = UnitreeG1AhGamepadTeleop(
            UnitreeG1AhGamepadTeleopConfig(initial_positions={"kBaseHeight.cmd": 0.6})
        )
        seeded.connect()
        assert seeded.get_action()["kBaseHeight.cmd"] == pytest.approx(0.6)
        seeded.disconnect()


def test_body_keys_never_change(teleop):
    default = default_action()
    body_keys = list(BODY_KEYS)

    layout = teleop.config.layout
    clock = {"t": 0.0}
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.buttons.add(layout.dpad_up)
        teleop.gamepad.buttons.add(layout.dpad_right)
        teleop.gamepad.buttons.add(layout.button_rb)
        teleop.gamepad.buttons.add(layout.button_lb)
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
    cfg = UnitreeG1AhGamepadTeleopConfig(initial_positions={"kHeadYaw.q": 0.3})
    with patch(f"{_MODULE}.UnitreeG1AhGamepadInput", FakeInput):
        t = UnitreeG1AhGamepadTeleop(cfg)
        t.connect()
        action = t.get_action()
        assert action["kHeadYaw.q"] == pytest.approx(0.3)
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


def test_default_config_uses_dualshock4_hidapi_preset():
    cfg = UnitreeG1AhGamepadTeleopConfig()
    assert cfg.preset == "dualshock4_hidapi"
    assert cfg.layout.button_rb == 10
    assert cfg.layout.hat is None
    assert cfg.layout.dpad_up == 11
    assert (cfg.layout.trigger_left, cfg.layout.trigger_right) == (4, 5)


def test_xbox_preset_gives_expected_layout():
    cfg = UnitreeG1AhGamepadTeleopConfig(preset="xbox")
    assert cfg.layout.button_rb == 5
    assert cfg.layout.hat == 0
    assert cfg.layout.dpad_up is None
    assert (cfg.layout.trigger_left, cfg.layout.trigger_right) == (2, 5)


def test_dualshock4_kernel_preset_gives_expected_layout():
    cfg = UnitreeG1AhGamepadTeleopConfig(preset="dualshock4_kernel")
    assert cfg.layout.button_rb == 5
    assert cfg.layout.button_y == 2
    assert (cfg.layout.trigger_left, cfg.layout.trigger_right) == (2, 5)


def test_explicit_layout_override_preserved_with_default_preset():
    cfg = UnitreeG1AhGamepadTeleopConfig(layout=GamepadLayout(button_rb=99))
    assert cfg.layout.button_rb == 99


def test_invalid_preset_raises():
    with pytest.raises(ValueError):
        UnitreeG1AhGamepadTeleopConfig(preset="not_a_real_preset")


def test_dpad_buttons_tilt_same_as_hat(teleop):
    clock = {"t": 0.0}
    layout = teleop.config.layout
    with patch(f"{_MODULE}.time.perf_counter", lambda: clock["t"]):
        teleop.get_action()
        teleop.gamepad.buttons.add(layout.dpad_up)
        for _ in range(200):
            clock["t"] += 0.05
            action = teleop.get_action()
        assert action["kHeadPitch.q"] == pytest.approx(0.8, abs=1e-6)


class _FakeJoystick:
    def __init__(self, numhats=1, hat_value=(0, 0), buttons=None):
        self._numhats = numhats
        self._hat_value = hat_value
        self._buttons = buttons or set()

    def get_numhats(self):
        return self._numhats

    def get_hat(self, index):
        return self._hat_value

    def get_button(self, index):
        return index in self._buttons


@skip_if_package_missing("pygame")
def test_hat_uses_dpad_buttons_when_configured():
    from lerobot.teleoperators.unitree_g1_ah_gamepad.gamepad_input import UnitreeG1AhGamepadInput

    layout = GamepadLayout.dualshock4_hidapi()
    gamepad = UnitreeG1AhGamepadInput(layout, deadzone=0.1)
    gamepad.joystick = _FakeJoystick(numhats=0, buttons={layout.dpad_right, layout.dpad_up})
    assert gamepad.hat() == (1, 1)


@skip_if_package_missing("pygame")
def test_hat_falls_back_to_hat_axis_when_no_dpad_buttons():
    from lerobot.teleoperators.unitree_g1_ah_gamepad.gamepad_input import UnitreeG1AhGamepadInput

    layout = GamepadLayout.dualshock4_kernel()
    gamepad = UnitreeG1AhGamepadInput(layout, deadzone=0.1)
    gamepad.joystick = _FakeJoystick(numhats=1, hat_value=(-1, 1))
    assert gamepad.hat() == (-1, 1)


def test_dpad_from_buttons_helper():
    layout = GamepadLayout.dualshock4_hidapi()

    class _Gamepad:
        def __init__(self, buttons):
            self._buttons = buttons

        def button(self, index):
            return index in self._buttons

    assert _dpad_from_buttons(_Gamepad({layout.dpad_left}), layout) == (-1, 0)
    assert _dpad_from_buttons(_Gamepad({layout.dpad_down}), layout) == (0, -1)
    assert _dpad_from_buttons(_Gamepad(set()), layout) == (0, 0)
