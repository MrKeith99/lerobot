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

"""Tests for `g1_ah_zmq`: client/server wire format and round trip over real sockets on localhost."""

import threading
import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("zmq")

from lerobot.robots.unitree_g1_ah import g1_ah_zmq as gz
from lerobot.robots.unitree_g1_ah.g1_ah_devices import HeadHandDevice
from lerobot.robots.unitree_g1_ah.g1_ah_joints import HEAD_HAND_MOTORS, HEAD_MOTORS
from tests.mocks.mock_unitree_g1_ah_server import MockHeadHandServer


def _wait_until(predicate, timeout_s: float = 3.0, interval_s: float = 0.01) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


@pytest.fixture
def mock_server():
    with MockHeadHandServer() as server:
        yield server


@pytest.fixture
def client(mock_server):
    c = gz.HeadHandZmqClient("127.0.0.1", state_port=mock_server.state_port, cmd_port=mock_server.cmd_port)
    c.connect()
    yield c
    c.disconnect()


def test_encode_decode_state_round_trip():
    payload = gz.encode_state({"a": 1, "b": 2}, torque=True, error=None)
    data = gz.decode_state(payload)
    assert data["ticks"] == {"a": 1, "b": 2}
    assert data["torque"] is True
    assert data["error"] is None


def test_encode_decode_cmd_round_trip():
    payload = gz.encode_cmd(goal_ticks={"a": 5}, torque=False)
    data = gz.decode_cmd(payload)
    assert data["goal_ticks"] == {"a": 5}
    assert data["torque"] is False


def test_client_read_latest_receives_all_motor_names(client):
    assert _wait_until(lambda: client.read_latest() is not None)
    state = client.read_latest()
    assert set(state["ticks"].keys()) == set(HEAD_HAND_MOTORS.keys())


def test_client_send_updates_server_ticks_and_records_cmd(client, mock_server):
    name = next(iter(HEAD_MOTORS))
    client.send(goal_ticks={name: 12345})

    assert _wait_until(lambda: mock_server.ticks[name] == 12345)
    assert any(c.get("goal_ticks", {}).get(name) == 12345 for c in mock_server.received)


def test_client_age_s_decreases_after_new_state(client):
    assert _wait_until(lambda: client.read_latest() is not None)
    first_age = client.age_s
    time.sleep(0.05)
    client.read_latest()
    second_age = client.age_s
    assert second_age is not None
    assert first_age is not None


def test_headhand_server_client_round_trip():
    head_bus = MagicMock(name="HeadBus")
    hand_bus = MagicMock(name="HandBus")
    head_bus.is_connected = True
    hand_bus.is_connected = True
    head_bus.sync_read.return_value = dict.fromkeys(HEAD_MOTORS, 2048)
    hand_bus.read.return_value = 512

    device = HeadHandDevice(
        "/dev/head",
        "/dev/hand",
        head_bus_cls=lambda **kw: head_bus,
        hand_bus_cls=lambda **kw: hand_bus,
    )

    server = gz.HeadHandServer(device, state_port=0, cmd_port=0, rate_hz=100.0)
    state_port, cmd_port = server.bind()
    shutdown = threading.Event()
    t = threading.Thread(target=server.run, args=(shutdown,), daemon=True)
    t.start()

    client = gz.HeadHandZmqClient("127.0.0.1", state_port=state_port, cmd_port=cmd_port)
    client.connect()
    try:
        assert _wait_until(lambda: client.read_latest() is not None)

        head_name = next(iter(HEAD_MOTORS))
        client.send(goal_ticks={head_name: 2200})
        assert _wait_until(lambda: head_bus.sync_write.called)
    finally:
        shutdown.set()
        t.join(timeout=2.0)
        client.disconnect()
        server.stop()


def test_headhand_server_keeps_running_on_bus_error():
    head_bus = MagicMock(name="HeadBus")
    hand_bus = MagicMock(name="HandBus")
    head_bus.is_connected = True
    hand_bus.is_connected = True

    call_count = {"n": 0}

    def flaky_sync_read(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("bus hiccup")
        return dict.fromkeys(HEAD_MOTORS, 2048)

    head_bus.sync_read.side_effect = flaky_sync_read
    hand_bus.read.return_value = 512

    device = HeadHandDevice(
        "/dev/head",
        "/dev/hand",
        head_bus_cls=lambda **kw: head_bus,
        hand_bus_cls=lambda **kw: hand_bus,
    )

    server = gz.HeadHandServer(device, state_port=0, cmd_port=0, rate_hz=100.0)
    state_port, cmd_port = server.bind()
    shutdown = threading.Event()
    t = threading.Thread(target=server.run, args=(shutdown,), daemon=True)
    t.start()

    client = gz.HeadHandZmqClient("127.0.0.1", state_port=state_port, cmd_port=cmd_port)
    client.connect()
    try:
        assert _wait_until(lambda: client.read_latest() is not None)
        got_error = _wait_until(lambda: (client.read_latest() or {}).get("error") is not None, timeout_s=1.0)
        got_recovered = _wait_until(lambda: (client.read_latest() or {}).get("ticks"), timeout_s=2.0)
        assert got_error or got_recovered
    finally:
        shutdown.set()
        t.join(timeout=2.0)
        client.disconnect()
        server.stop()
