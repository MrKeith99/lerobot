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

"""Tests for `headhand_devices`: tick/rad conversion and `HeadHandDevice` bus wiring. No hardware."""

import logging
from unittest.mock import MagicMock

import pytest

from lerobot.robots.unitree_g1 import headhand_devices as d
from lerobot.robots.unitree_g1.end_effectors import AMAZING_HAND_LIMIT_RAD, AMAZING_HAND_MOTORS as HAND_MOTORS
from lerobot.robots.unitree_g1.hand_force import HandForceLimiter
from lerobot.robots.unitree_g1.heads import HEAD_LIMITS_RAD, HEAD_MOTORS


def _bus_factory(motors_seen: dict) -> MagicMock:
    bus = MagicMock(name="BusMock")
    bus.is_connected = False

    def _connect(handshake=True):
        bus.is_connected = True

    def _disconnect(disable_torque=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    return bus


@pytest.fixture
def head_bus_mock():
    return _bus_factory({})


@pytest.fixture
def hand_bus_mock():
    return _bus_factory({})


@pytest.fixture
def device(head_bus_mock, hand_bus_mock):
    head_kwargs = {}
    hand_kwargs = {}

    def head_bus_cls(**kwargs):
        head_kwargs.update(kwargs)
        head_bus_mock.motors = kwargs["motors"]
        return head_bus_mock

    def hand_bus_cls(**kwargs):
        hand_kwargs.update(kwargs)
        hand_bus_mock.motors = kwargs["motors"]
        return hand_bus_mock

    dev = d.HeadHandDevice("/dev/head", "/dev/hand", head_bus_cls=head_bus_cls, hand_bus_cls=hand_bus_cls)
    dev._head_kwargs = head_kwargs
    dev._hand_kwargs = hand_kwargs
    return dev


def test_ticks_to_rad_zero_tick_is_zero_rad():
    calib = d.default_calibration("kHeadYaw")
    assert d.ticks_to_rad("xl330-m288", calib.homing_offset, calib) == pytest.approx(0.0)


def test_ticks_to_rad_drive_mode_flips_sign():
    calib_normal = d.default_calibration("kHeadYaw")
    tick = calib_normal.homing_offset + 100
    positive = d.ticks_to_rad("xl330-m288", tick, calib_normal)

    from lerobot.motors.motors_bus import MotorCalibration

    calib_inverted = MotorCalibration(
        id=calib_normal.id,
        drive_mode=1,
        homing_offset=calib_normal.homing_offset,
        range_min=calib_normal.range_min,
        range_max=calib_normal.range_max,
    )
    negative = d.ticks_to_rad("xl330-m288", tick, calib_inverted)
    assert negative == pytest.approx(-positive)


def test_rad_to_ticks_clamps_at_calibration_range():
    from lerobot.motors.motors_bus import MotorCalibration

    calib = MotorCalibration(id=1, drive_mode=0, homing_offset=2048, range_min=2000, range_max=2100)
    assert d.rad_to_ticks("xl330-m288", 10.0, calib) == 2100
    assert d.rad_to_ticks("xl330-m288", -10.0, calib) == 2000


def test_rad_to_ticks_clamps_at_tick_range():
    from lerobot.motors.motors_bus import MotorCalibration

    calib = MotorCalibration(id=1, drive_mode=0, homing_offset=2048, range_min=0, range_max=4095)
    assert d.rad_to_ticks("xl330-m288", 100.0, calib) == 4095
    assert d.rad_to_ticks("xl330-m288", -100.0, calib) == 0


@pytest.mark.parametrize("model,name", [("xl330-m288", "kHeadYaw"), ("scs0009", "kRightHandMotor1")])
def test_rad_to_ticks_round_trip(model, name):
    calib = d.default_calibration(name)
    for tick in (calib.homing_offset - 50, calib.homing_offset, calib.homing_offset + 50):
        rad = d.ticks_to_rad(model, tick, calib)
        assert d.rad_to_ticks(model, rad, calib) == tick


def test_clamp_rad_head_limits():
    low, high = HEAD_LIMITS_RAD["kHeadYaw"]
    assert d.clamp_rad("kHeadYaw", low - 1.0) == pytest.approx(low)
    assert d.clamp_rad("kHeadYaw", high + 1.0) == pytest.approx(high)
    assert d.clamp_rad("kHeadYaw", 0.0) == pytest.approx(0.0)


def test_clamp_rad_hand_limits():
    name = next(iter(HAND_MOTORS))
    assert d.clamp_rad(name, 100.0) == pytest.approx(AMAZING_HAND_LIMIT_RAD)
    assert d.clamp_rad(name, -100.0) == pytest.approx(-AMAZING_HAND_LIMIT_RAD)


def test_clamp_rad_unknown_name_raises():
    with pytest.raises(ValueError):
        d.clamp_rad("not_a_motor", 0.0)


def test_head_hand_device_bus_construction(device):
    assert device._hand_kwargs["protocol_version"] == 1
    assert len(device._hand_kwargs["motors"]) == 16
    assert len(device._head_kwargs["motors"]) == 2


def test_read_ticks_uses_sync_read_on_head_and_read_on_hands(device, head_bus_mock, hand_bus_mock):
    head_bus_mock.sync_read.return_value = dict.fromkeys(HEAD_MOTORS, 2048)
    hand_bus_mock.read.return_value = 512

    ticks = device.read_ticks()

    head_bus_mock.sync_read.assert_called_once_with("Present_Position", normalize=False)
    assert hand_bus_mock.read.call_count == 16
    assert hand_bus_mock.sync_read.call_count == 0
    for call in hand_bus_mock.read.call_args_list:
        assert call.kwargs["normalize"] is False
    assert len(ticks) == 18


def test_write_ticks_clamps_and_syncs_per_bus(device, head_bus_mock, hand_bus_mock):
    head_name = next(iter(HEAD_MOTORS))
    hand_name = next(iter(HAND_MOTORS))
    goals = {head_name: 999999, hand_name: -999999}

    device.write_ticks(goals)

    head_bus_mock.sync_write.assert_called_once()
    args, kwargs = head_bus_mock.sync_write.call_args
    assert args[0] == "Goal_Position"
    assert args[1][head_name] == d.TICK_RANGE["xl330-m288"][1]
    assert kwargs["normalize"] is False

    hand_bus_mock.sync_write.assert_called_once()
    args, kwargs = hand_bus_mock.sync_write.call_args
    assert args[1][hand_name] == d.TICK_RANGE["scs0009"][0]
    assert kwargs["normalize"] is False


def test_ping_all_maps_exceptions_to_none(device, head_bus_mock, hand_bus_mock):
    head_bus_mock.ping.side_effect = ConnectionError("nope")
    hand_bus_mock.ping.return_value = 1234

    results = device.ping_all()

    for name in HEAD_MOTORS:
        assert results[name] is None
    for name in HAND_MOTORS:
        assert results[name] == 1234


def test_set_torque_enables_and_disables_both_buses(device, head_bus_mock, hand_bus_mock):
    device.set_torque(True)
    head_bus_mock.enable_torque.assert_called_once()
    hand_bus_mock.enable_torque.assert_called_once()

    device.set_torque(False)
    head_bus_mock.disable_torque.assert_called_once()
    hand_bus_mock.disable_torque.assert_called_once()


def test_head_only_device_serves_the_head_motors(head_bus_mock):
    head_bus_mock.sync_read.return_value = {"kHeadYaw": 2000, "kHeadPitch": 2100}
    dev = d.HeadHandDevice("/dev/head", None, head_bus_cls=lambda **kwargs: head_bus_mock)
    assert dev.hand_bus is None
    assert set(dev.models) == set(d.HEAD_MOTORS)
    dev.connect()
    assert dev.read_ticks() == {"kHeadYaw": 2000, "kHeadPitch": 2100}
    dev.write_ticks({"kHeadYaw": 2050, "kRightHandMotor1": 500})
    head_bus_mock.sync_write.assert_called_once_with("Goal_Position", {"kHeadYaw": 2050}, normalize=False)
    assert set(dev.ping_all()) == set(d.HEAD_MOTORS)


def test_hands_only_device_serves_the_hand_motors(hand_bus_mock):
    hand_bus_mock.read.return_value = 512
    dev = d.HeadHandDevice(None, "/dev/hand", hand_bus_cls=lambda **kwargs: hand_bus_mock)
    assert dev.head_bus is None
    dev.connect()
    assert set(dev.read_ticks()) == set(d.HAND_MOTORS)
    dev.set_torque(False)
    hand_bus_mock.disable_torque.assert_called_once()
    dev.write_ticks({"kHeadYaw": 2050})
    hand_bus_mock.sync_write.assert_not_called()


def test_device_needs_a_bus():
    with pytest.raises(ValueError):
        d.HeadHandDevice(None, None)


def _limited_device(head_bus_mock, hand_bus_mock, limiter=None, **kwargs):
    def head_bus_cls(**kw):
        return head_bus_mock

    def hand_bus_cls(**kw):
        return hand_bus_mock

    return d.HeadHandDevice(
        "/dev/head",
        "/dev/hand",
        head_bus_cls=head_bus_cls,
        hand_bus_cls=hand_bus_cls,
        hand_force=limiter,
        **kwargs,
    )


def _fake_hand_reads(hand_bus_mock, pos=500, load=0, temp=30):
    values = {"Present_Position": pos, "Present_Load": load, "Present_Temperature": temp}
    hand_bus_mock.read.side_effect = lambda reg, name, **kw: values[reg]
    return values


def test_without_limiter_reads_only_positions_and_writes_goals_untouched(head_bus_mock, hand_bus_mock):
    dev = _limited_device(head_bus_mock, hand_bus_mock)
    _fake_hand_reads(hand_bus_mock)
    head_bus_mock.sync_read.return_value = dict.fromkeys(HEAD_MOTORS, 2048)
    dev.read_ticks()
    assert {c.args[0] for c in hand_bus_mock.read.call_args_list} == {"Present_Position"}
    name = next(iter(HAND_MOTORS))
    dev.write_ticks({name: 900})
    assert hand_bus_mock.sync_write.call_args.args[1] == {name: 900}


def test_limiter_reads_loads_and_temperature_every_n_cycles(head_bus_mock, hand_bus_mock):
    dev = _limited_device(head_bus_mock, hand_bus_mock, HandForceLimiter(), temp_every=3)
    _fake_hand_reads(hand_bus_mock)
    head_bus_mock.sync_read.return_value = dict.fromkeys(HEAD_MOTORS, 2048)
    counts = []
    for _ in range(4):
        hand_bus_mock.read.reset_mock()
        ticks = dev.read_ticks()
        regs = [c.args[0] for c in hand_bus_mock.read.call_args_list]
        counts.append(
            (regs.count("Present_Position"), regs.count("Present_Load"), regs.count("Present_Temperature"))
        )
        assert len(ticks) == 18
    assert counts == [(16, 16, 16), (16, 16, 0), (16, 16, 0), (16, 16, 16)]


def test_limiter_caps_hand_goals_but_not_head_goals(head_bus_mock, hand_bus_mock):
    dev = _limited_device(head_bus_mock, hand_bus_mock, HandForceLimiter())
    _fake_hand_reads(hand_bus_mock, pos=500)
    head_bus_mock.sync_read.return_value = dict.fromkeys(HEAD_MOTORS, 2048)
    dev.read_ticks()
    head_name, hand_name = next(iter(HEAD_MOTORS)), next(iter(HAND_MOTORS))
    dev.write_ticks({head_name: 3000, hand_name: 900})
    assert head_bus_mock.sync_write.call_args.args[1] == {head_name: 3000}
    assert hand_bus_mock.sync_write.call_args.args[1] == {hand_name: 530}


def test_limiter_overload_logs_warning_and_backs_off(head_bus_mock, hand_bus_mock, caplog, monkeypatch):
    clock = iter(range(1000))
    monkeypatch.setattr(d.time, "monotonic", lambda: next(clock) * 0.1)
    dev = _limited_device(head_bus_mock, hand_bus_mock, HandForceLimiter())
    _fake_hand_reads(hand_bus_mock, pos=500, load=800)
    head_bus_mock.sync_read.return_value = dict.fromkeys(HEAD_MOTORS, 2048)
    dev.read_ticks()
    name = next(iter(HAND_MOTORS))
    with caplog.at_level(logging.INFO):
        goals = [
            dev.write_ticks({name: 900}) or hand_bus_mock.sync_write.call_args.args[1][name] for _ in range(5)
        ]
    assert goals[-1] == 485
    assert any(r.levelno == logging.WARNING and "backing off" in r.message for r in caplog.records)


def test_limiter_temperature_torque_off_and_back_on(head_bus_mock, hand_bus_mock):
    dev = _limited_device(head_bus_mock, hand_bus_mock, HandForceLimiter(), temp_every=1)
    values = _fake_hand_reads(hand_bus_mock, pos=500, temp=70)
    head_bus_mock.sync_read.return_value = dict.fromkeys(HEAD_MOTORS, 2048)
    dev.set_torque(True)
    hand_bus_mock.enable_torque.reset_mock()
    dev.read_ticks()
    name = next(iter(HAND_MOTORS))
    dev.write_ticks({name: 900})
    hand_bus_mock.disable_torque.assert_called_once_with(name)
    hand_bus_mock.sync_write.assert_not_called()

    dev.set_torque(True)
    assert name not in hand_bus_mock.enable_torque.call_args.args[0]

    values["Present_Temperature"] = 50
    dev.read_ticks()
    dev.write_ticks({name: 900})
    hand_bus_mock.enable_torque.assert_called_with(name)
    assert hand_bus_mock.sync_write.call_args.args[1] == {name: 530}


def test_read_hand_feedback_returns_pos_load_temp(head_bus_mock, hand_bus_mock):
    dev = _limited_device(head_bus_mock, hand_bus_mock)
    _fake_hand_reads(hand_bus_mock, pos=512, load=1024 | 40, temp=33)
    feedback = dev.read_hand_feedback()
    assert set(feedback) == set(HAND_MOTORS)
    assert feedback[next(iter(HAND_MOTORS))] == (512, -40, 33)
