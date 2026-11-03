#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import contextlib
import json
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.unitree_g1.safety import DAMPING_KD, LowCmdWatchdog, damping_motor_cmds
from lerobot.utils.import_utils import _unitree_sdk_available


def test_damping_commands_have_no_stiffness_and_no_torque():
    cmds = damping_motor_cmds()
    assert len(cmds) == 29
    assert all(c["kp"] == 0.0 and c["tau"] == 0.0 and c["mode"] == 1 for c in cmds)
    assert [c["kd"] for c in cmds] == list(DAMPING_KD)
    assert min(DAMPING_KD) > 0.0


def test_damping_rejects_wrong_length():
    with pytest.raises(ValueError):
        damping_motor_cmds([1.0] * 3)


def test_watchdog_starts_in_damping_and_forwards_a_client():
    wd = LowCmdWatchdog(timeout_s=0.3)
    assert wd.check(0.0) is True
    assert wd.on_command("a", 1.0) is True
    assert wd.check(1.2) is False


def test_watchdog_latches_damping_for_a_timed_out_session():
    wd = LowCmdWatchdog(timeout_s=0.3)
    wd.on_command("a", 1.0)
    assert wd.check(1.31) is True
    assert wd.on_command("a", 1.4) is False
    assert wd.check(1.5) is True


def test_watchdog_accepts_a_new_session_after_a_trip():
    wd = LowCmdWatchdog(timeout_s=0.3)
    wd.on_command("a", 1.0)
    wd.check(2.0)
    assert wd.on_command("b", 2.1) is True
    assert wd.check(2.2) is False


def test_watchdog_ignores_a_second_session_while_one_is_active():
    wd = LowCmdWatchdog(timeout_s=0.3)
    wd.on_command("a", 1.0)
    assert wd.on_command("b", 1.1) is False
    assert wd.session == "a"


# ---------------------------------------------------------------------------
# Relay command loop and robot client (need the Unitree SDK)
# ---------------------------------------------------------------------------

sdk = pytest.mark.skipif(not _unitree_sdk_available, reason="Unitree SDK not available")


@contextlib.contextmanager
def _relay_loop(timeout_s=0.1, handover=None):
    import zmq

    from lerobot.robots.unitree_g1 import run_g1_server

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    push = ctx.socket(zmq.PUSH)
    push.connect(f"tcp://127.0.0.1:{port}")
    written = []
    pub = MagicMock()
    pub.Write.side_effect = lambda cmd: written.append(
        [(cmd.motor_cmd[i].kp, cmd.motor_cmd[i].kd, cmd.motor_cmd[i].q) for i in range(29)]
    )
    stop = threading.Event()
    thread = threading.Thread(
        target=run_g1_server.cmd_forward_loop,
        args=(pull, pub, MagicMock(), LowCmdWatchdog(timeout_s), 4),
        kwargs={"shutdown_event": stop, "handover": handover},
        daemon=True,
    )
    thread.start()

    def send(session, kp=50.0, q=0.5, handover=False):
        cmd = {"mode": 1, "q": q, "dq": 0.0, "kp": kp, "kd": 1.0, "tau": 0.0}
        msg = {"topic": "rt/lowcmd", "data": {"motor_cmd": [cmd] * 29}, "session": session}
        if handover:
            msg["handover"] = True
        push.send(json.dumps(msg).encode())

    try:
        yield send, written
    finally:
        stop.set()
        thread.join(timeout=2.0)
        push.close(linger=0)
        pull.close(linger=0)
        ctx.term()


def _is_damping(cmd):
    return all(kp == 0.0 for kp, _, _ in cmd) and [kd for _, kd, _ in cmd] == pytest.approx(list(DAMPING_KD))


@sdk
def test_relay_damps_before_a_client_and_after_it_goes_silent():
    with _relay_loop(timeout_s=0.1) as (send, written):
        time.sleep(0.1)
        assert written and all(_is_damping(c) for c in written)
        for _ in range(5):
            send("s1")
            time.sleep(0.02)
        time.sleep(0.05)
        assert any(c[0][0] == 50.0 for c in written)
        time.sleep(0.3)
        assert _is_damping(written[-1])
        n = len(written)
        send("s1")  # the stalled session cannot take the robot back
        time.sleep(0.1)
        assert all(_is_damping(c) for c in written[n:])
        send("s2")  # a freshly connected client can
        time.sleep(0.05)
        assert any(c[0][0] == 50.0 for c in written[n:])


@contextlib.contextmanager
def _real_robot(**config_kwargs):
    from tests.robots.test_unitree_g1 import _mocked_unitree_g1

    with _mocked_unitree_g1(**config_kwargs) as (robot, mocks):
        robot.config.is_simulation = False
        robot._ChannelFactoryInitialize = MagicMock()
        robot._ChannelPublisher = MagicMock(return_value=mocks["publisher_mock"])
        robot._ChannelSubscriber = MagicMock(return_value=mocks["subscriber_mock"])
        mocks["lowstate_msg"].relay_state = "damping"
        written = []
        mocks["publisher_mock"].Write.side_effect = lambda msg, handover=False: written.append(
            [(float(msg.motor_cmd[i].kp), float(msg.motor_cmd[i].kd)) for i in range(29)]
        )
        yield robot, mocks, written


def _all_damping(cmd):
    return all(kp == 0.0 for kp, _ in cmd) and [kd for _, kd in cmd] == pytest.approx(list(DAMPING_KD))


@sdk
def test_real_disconnect_sends_damping_last_never_zero_torque():
    with _real_robot() as (robot, _, written):
        robot.connect(calibrate=False)
        robot.send_action({"kLeftElbow.q": 0.2})
        time.sleep(0.05)
        robot.disconnect()
        assert _all_damping(written[-1])
        assert not any(all(kp == 0.0 and kd == 0.0 for kp, kd in cmd) for cmd in written)


@sdk
def test_real_disconnect_with_a_controller_damps_after_the_controller_stopped():
    with _real_robot(controller="GrootLocomotionController", locomotion_start_gate=False) as (
        robot,
        _,
        written,
    ):
        robot.controller.run_step.return_value = {"kLeftHipPitch.q": 0.1}
        robot.controller.kp = np.full(29, 100.0)
        robot.controller.kd = np.full(29, 2.0)
        robot.connect(calibrate=False)
        time.sleep(0.1)
        assert any(cmd[0][0] == 100.0 for cmd in written)
        robot.disconnect()
        time.sleep(0.05)
        assert _all_damping(written[-1])


@sdk
def test_real_keepalive_republishes_once_commanded():
    with _real_robot() as (robot, _, written):
        robot.connect(calibrate=False)
        time.sleep(0.06)
        assert written == []  # nothing until the first action: the relay holds damping
        robot.send_action({"kLeftElbow.q": 0.2})
        n = len(written)
        time.sleep(0.1)
        assert len(written) - n >= 3


@sdk
def test_real_stale_lowstate_damps_and_fails_the_next_call():
    with _real_robot(lowstate_timeout_s=0.05) as (robot, mocks, written):
        robot.connect(calibrate=False)
        robot.send_action({"kLeftElbow.q": 0.2})
        mocks["subscriber_mock"].Read.return_value = None
        time.sleep(0.2)
        assert robot._fault is not None
        assert _all_damping(written[-1])
        with pytest.raises(RuntimeError, match="damping"):
            robot.send_action({"kLeftElbow.q": 0.3})
        with pytest.raises(RuntimeError, match="damping"):
            robot.get_observation()
        robot.publish_lowcmd({"kLeftElbow.q": 0.3})
        assert _all_damping(written[-1])


@sdk
def test_sim_disconnect_sends_no_damping():
    from tests.robots.test_unitree_g1 import _mocked_unitree_g1

    with _mocked_unitree_g1() as (robot, mocks):
        robot.msg = MagicMock()
        robot.disconnect()
        mocks["publisher_mock"].Write.assert_not_called()


def _stream(send, session, n=5, **kw):
    for _ in range(n):
        send(session, **kw)
        time.sleep(0.02)


@sdk
def test_relay_handover_holds_then_stops_publishing():
    from lerobot.robots.unitree_g1.handover import Handover, HandoverState

    def give_back(stop_publishing):
        time.sleep(0.4)  # longer than the watchdog timeout: the hold must not turn into damping
        stop_publishing()
        return None

    handover = Handover(give_back)
    with _relay_loop(timeout_s=0.1, handover=handover) as (send, written):
        _stream(send, "s1")
        send("s1", kp=60.0, handover=True)
        time.sleep(0.2)
        assert handover.state == HandoverState.HOLDING
        assert written[-1][0][0] == 60.0
        time.sleep(0.4)
        assert handover.state == HandoverState.DONE
        n = len(written)
        send("s1")
        send("s2")
        time.sleep(0.3)
        assert len(written) == n  # Unitree's controller owns the robot: nothing more, not even damping
        assert not any(_is_damping(c) for c in written[-20:])


@sdk
def test_relay_failed_handover_keeps_holding_until_the_client_damps():
    from lerobot.robots.unitree_g1.handover import Handover, HandoverState

    handover = Handover(lambda stop_publishing: "firmware said no")
    with _relay_loop(timeout_s=0.1, handover=handover) as (send, written):
        _stream(send, "s1")
        send("s1", kp=60.0, handover=True)
        time.sleep(0.4)
        assert handover.state == HandoverState.FAILED
        assert all(c[0][0] == 60.0 for c in written[-10:])  # still holding, no damping
        send("s1", kp=0.0)
        time.sleep(0.05)
        assert handover.state == HandoverState.CANCELLED
        time.sleep(0.3)
        assert _is_damping(written[-1])  # client silent again: the watchdog damps


@sdk
def test_relay_without_handover_strategy_forwards_and_reports_unsupported():
    from lerobot.robots.unitree_g1.handover import Handover, HandoverState

    handover = Handover(None)
    with _relay_loop(timeout_s=0.1, handover=handover) as (send, written):
        _stream(send, "s1")
        send("s1", kp=60.0, handover=True)
        time.sleep(0.03)
        assert handover.state == HandoverState.UNSUPPORTED
        assert any(c[0][0] == 60.0 for c in written)
        time.sleep(0.3)
        assert _is_damping(written[-1])


@sdk
def test_real_handover_on_exit_holds_stiff_until_the_relay_confirms():
    with _real_robot(handover_on_exit=True) as (robot, mocks, written):
        flags = []
        write = mocks["publisher_mock"].Write.side_effect

        def capture(msg, handover=False):
            flags.append(handover)
            write(msg)
            if sum(flags) == 5:
                mocks["lowstate_msg"].relay_handover = "done"

        mocks["publisher_mock"].Write.side_effect = capture
        mocks["lowstate_msg"].relay_handover = "holding"
        robot.connect(calibrate=False)
        robot.send_action({"kLeftElbow.q": 0.2})
        robot.disconnect()
        assert sum(flags) >= 5
        assert flags[-1] is True
        assert written[-1][0][0] == robot.kp[0] > 0  # legs stiff, never damping or zero torque
        assert not any(_all_damping(c) for c in written)


@sdk
def test_real_handover_refused_holds_then_damps_on_a_fault():
    with _real_robot(handover_on_exit=True, lowstate_timeout_s=0.05) as (robot, mocks, written):
        mocks["lowstate_msg"].relay_handover = "unsupported"
        robot.connect(calibrate=False)
        robot.send_action({"kLeftElbow.q": 0.2})
        threading.Timer(0.2, lambda: setattr(mocks["subscriber_mock"].Read, "return_value", None)).start()
        robot.disconnect()
        assert written[-2][0][0] == robot.kp[0]  # held stiffly first
        assert _all_damping(written[-1])


@sdk
@pytest.mark.parametrize("state", [None, "forwarding", "handover", "handed_over"])
def test_real_connect_refuses_unless_the_relay_reports_damping(state):
    with _real_robot() as (robot, mocks, written):
        mocks["lowstate_msg"].relay_state = state
        with pytest.raises(RuntimeError, match="not confirmed in damping"):
            robot.connect(calibrate=False)
        time.sleep(0.05)
        assert written == []
        assert not robot.subscribe_thread.is_alive()


@sdk
def test_real_connect_check_can_be_turned_off():
    with _real_robot(require_damping_at_connect=False) as (robot, mocks, _):
        mocks["lowstate_msg"].relay_state = None
        robot.connect(calibrate=False)
        assert robot.is_connected


def test_check_damping_needs_fsm_damp_or_an_explicit_assumption():
    from lerobot.robots.unitree_g1.handover import FSM_DAMP, check_damping

    assert check_damping(FSM_DAMP, assume_damping=False) is None
    assert "not damping" in check_damping(0, assume_damping=False)
    assert "not damping" in check_damping(500, assume_damping=True)
    assert "--assume-damping" in check_damping(None, assume_damping=False)
    assert check_damping(None, assume_damping=True) is None


def test_relay_state_reports_what_the_relay_does():
    from lerobot.robots.unitree_g1.handover import Handover, HandoverState, relay_state

    h = Handover(None)
    assert relay_state(False, h) == "damping"
    assert relay_state(True, h) == "forwarding"
    h.state = HandoverState.HOLDING
    assert relay_state(True, h) == "handover"
    h.state = HandoverState.DONE
    assert relay_state(False, h) == "handed_over"
