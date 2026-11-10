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
    goals = {head_name: 9_999_999, hand_name: -999999}

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
    assert hand_bus_mock.disable_torque.call_count == len(HAND_MOTORS)


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
    assert hand_bus_mock.disable_torque.call_count == len(HAND_MOTORS)
    hand_bus_mock.sync_write.reset_mock()
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
        **{"hand_block_read": False, **kwargs},
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
    dev = _limited_device(
        head_bus_mock, hand_bus_mock, HandForceLimiter(max_err_ticks=30, load_limit=500), temp_every=3
    )
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
    dev = _limited_device(head_bus_mock, hand_bus_mock, HandForceLimiter(max_err_ticks=30, load_limit=500))
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
    dev = _limited_device(head_bus_mock, hand_bus_mock, HandForceLimiter(max_err_ticks=30, load_limit=500))
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
    dev = _limited_device(
        head_bus_mock, hand_bus_mock, HandForceLimiter(max_err_ticks=30, load_limit=500), temp_every=1
    )
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


def test_hand_torque_off_survives_error_replies_and_missing_servos(hand_bus_mock, caplog):
    """One servo replying with an error flag (or a missing hand) must not leave the other servos powered."""
    dev = d.HeadHandDevice(None, "/dev/hand", hand_bus_cls=lambda **kwargs: hand_bus_mock)
    dev.connect()
    bad = {"kRightHandMotor3", "kLeftHandMotor11"}

    def _disable(name):
        if name in bad:
            raise RuntimeError("[RxPacketError] Input voltage error!")

    hand_bus_mock.disable_torque.side_effect = _disable
    with caplog.at_level(logging.WARNING):
        dev.set_torque(False)
    hand_bus_mock.sync_write.assert_called_once_with(
        "Torque_Enable", dict.fromkeys(HAND_MOTORS, 0), normalize=False
    )
    assert [c.args[0] for c in hand_bus_mock.disable_torque.call_args_list] == list(HAND_MOTORS)
    assert sum("Input voltage error" in r.message for r in caplog.records) == len(bad)


def test_hand_disconnect_with_torque_off_uses_the_safe_path(hand_bus_mock):
    dev = d.HeadHandDevice(None, "/dev/hand", hand_bus_cls=lambda **kwargs: hand_bus_mock)
    dev.connect()
    hand_bus_mock.sync_write.side_effect = ConnectionError("port gone")
    dev.disconnect(disable_torque=True)
    assert hand_bus_mock.disable_torque.call_count == len(HAND_MOTORS)
    hand_bus_mock.disconnect.assert_called_once_with(disable_torque=False)


# Head in mode 5 (multi-turn, Goal_Current cap) with a software box in mode-5 ticks (lab G1, 2026-11-10: yaw
# 1734..2831, pitch -523..648 with 0 = facing front). The turn count restarts in 0..4095 at power-up.
LAB_BOX = {"kHeadYaw": (1734, 2831), "kHeadPitch": (-523, 648)}


def _configured_head(present: dict[str, int]):
    bus = _bus_factory({})
    bus.sync_read.side_effect = lambda *a, **kw: dict(present)
    dev = d.HeadHandDevice("/dev/head", None, head_bus_cls=lambda **kw: bus, head_box=LAB_BOX)
    dev.connect()
    dev.configure()
    return dev, bus


def test_configure_sets_mode_5_and_goal_current_never_the_current_limit():
    dev, bus = _configured_head({"kHeadYaw": 2282, "kHeadPitch": 600})
    writes = {(c.args[0], c.args[1], c.args[2]) for c in bus.write.call_args_list}
    for name in HEAD_MOTORS:
        assert ("Operating_Mode", name, 5) in writes
        assert ("Goal_Current", name, d.HEAD_GOAL_CURRENT) in writes
    assert not any(c.args[0] == "Current_Limit" for c in bus.write.call_args_list)
    assert dev.head_limits == LAB_BOX
    # torque stays off after configure (only set_torque powers the head), and no EEPROM setup on the shared head
    bus.disable_torque.assert_called_once_with()
    bus.enable_torque.assert_not_called()
    bus.configure_motors.assert_not_called()


def test_head_inside_box_holds_its_position_then_gets_torque():
    dev, bus = _configured_head({"kHeadYaw": 2282, "kHeadPitch": 600})
    dev.set_torque(True)
    bus.sync_write.assert_called_once_with(
        "Goal_Position", {"kHeadYaw": 2282, "kHeadPitch": 600}, normalize=False
    )
    assert set(bus.enable_torque.call_args.args[0]) == {"kHeadYaw", "kHeadPitch"}
    assert dev.fault is None


def test_pitch_resting_on_its_down_stop_gets_torque_and_is_clamped_into_the_box():
    dev, bus = _configured_head({"kHeadYaw": 2282, "kHeadPitch": 720})  # 72 past the box: within tolerance
    dev.set_torque(True)
    assert set(bus.enable_torque.call_args.args[0]) == {"kHeadYaw", "kHeadPitch"}
    bus.sync_write.reset_mock()
    dev.write_ticks({"kHeadPitch": 900})
    bus.sync_write.assert_called_once_with("Goal_Position", {"kHeadPitch": 648}, normalize=False)


def test_pitch_looking_up_at_power_up_is_unwrapped_by_one_turn():
    dev, bus = _configured_head({"kHeadYaw": 2282, "kHeadPitch": 3800})  # = -296 (looking up)
    assert dev.read_ticks()["kHeadPitch"] == -296
    dev.set_torque(True)
    bus.sync_write.assert_called_once_with(
        "Goal_Position", {"kHeadYaw": 2282, "kHeadPitch": 3800}, normalize=False
    )
    bus.sync_write.reset_mock()
    dev.write_ticks({"kHeadPitch": -400, "kHeadYaw": 9999})
    bus.sync_write.assert_called_once_with(
        "Goal_Position", {"kHeadPitch": -400 + 4096, "kHeadYaw": 2831}, normalize=False
    )


def test_head_far_outside_its_box_keeps_torque_off_and_ignores_goals(caplog):
    present = {"kHeadYaw": 1200, "kHeadPitch": 600}  # yaw 534 ticks right of the box
    dev, bus = _configured_head(present)
    with caplog.at_level(logging.ERROR):
        dev.set_torque(True)
    bus.enable_torque.assert_called_once_with(["kHeadPitch"])
    assert "kHeadYaw reads 1200" in dev.fault
    assert "kHeadYaw reads 1200" in caplog.text

    bus.sync_write.reset_mock()
    dev.write_ticks({"kHeadYaw": 2000, "kHeadPitch": 100})
    bus.sync_write.assert_called_once_with("Goal_Position", {"kHeadPitch": 100}, normalize=False)

    present["kHeadYaw"] = 2000  # turned back by hand
    bus.enable_torque.reset_mock()
    dev.set_torque(True)
    assert set(bus.enable_torque.call_args.args[0]) == {"kHeadYaw", "kHeadPitch"}
    assert dev.fault is None


def test_head_without_configure_keeps_the_old_behaviour(device, head_bus_mock):
    device.set_torque(True)
    head_bus_mock.enable_torque.assert_called_once_with()
    assert device.fault is None


def _fake_block_reads(hand_bus_mock, pos=500, raw_load=0, temp=30):
    hand_bus_mock.motors = d.build_hand_motors()
    reply = [pos >> 8, pos & 0xFF, 0, 0, raw_load >> 8, raw_load & 0xFF, 60, temp]
    hand_bus_mock.packet_handler.readTxRx.return_value = (reply, 0, 0)


def test_block_read_decodes_big_endian_scs_registers(head_bus_mock, hand_bus_mock):
    # Bytes as read on the robot 2026-11-10: position 463 = 0x01CF, load 0x0400 | 300 = -300 per mille.
    dev = _limited_device(
        head_bus_mock, hand_bus_mock, HandForceLimiter(max_err_ticks=30, load_limit=500), hand_block_read=True
    )
    _fake_block_reads(hand_bus_mock, pos=463, raw_load=0x0400 | 300, temp=24)
    head_bus_mock.sync_read.return_value = dict.fromkeys(HEAD_MOTORS, 2048)
    ticks = dev.read_ticks()
    name = next(iter(HAND_MOTORS))
    assert ticks[name] == 463
    assert dev._hand_load[name] == -300
    assert dev._hand_temp[name] == 24
    assert hand_bus_mock.packet_handler.readTxRx.call_count == 16
    assert hand_bus_mock.packet_handler.readTxRx.call_args.args[2:] == (d.SCS_BLOCK_ADDR, 8)
    hand_bus_mock.read.assert_not_called()


def test_block_read_retries_once_then_raises(head_bus_mock, hand_bus_mock):
    dev = _limited_device(
        head_bus_mock, hand_bus_mock, HandForceLimiter(max_err_ticks=30, load_limit=500), hand_block_read=True
    )
    _fake_block_reads(hand_bus_mock)
    good = hand_bus_mock.packet_handler.readTxRx.return_value
    hand_bus_mock.packet_handler.readTxRx.side_effect = [([], -6, 0), good] + [good] * 15
    dev.read_ticks()  # one failed reply is retried
    hand_bus_mock.packet_handler.readTxRx.side_effect = None
    hand_bus_mock.packet_handler.readTxRx.return_value = ([], -6, 0)
    with pytest.raises(ConnectionError, match="block read failed"):
        dev.read_ticks()


def test_head_resting_on_its_down_stop_gets_torque(head_bus_mock, hand_bus_mock):
    # Torque off on exit lets the head sink to pitch ~813 (2026-11-10), past the box's 648: still allowed.
    dev = _limited_device(head_bus_mock, hand_bus_mock)
    dev.head_limits = {"kHeadYaw": (1734, 2831), "kHeadPitch": (-523, 648)}
    dev.head_turns = dict.fromkeys(dev.head_limits, 0)
    head_bus_mock.sync_read.return_value = {"kHeadYaw": 2290, "kHeadPitch": 813}
    dev.set_torque(True)
    assert dev.head_faults == {}
    head_bus_mock.enable_torque.assert_called_with(["kHeadYaw", "kHeadPitch"])
