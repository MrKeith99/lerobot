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

"""ZMQ transport for the UnitreeG1Ah head/hand bridge, mirroring `unitree_g1`'s lowcmd/lowstate pattern.

Ports and topics are separate from the body's `LOWCMD_PORT`/`LOWSTATE_PORT` so the head/hand
bridge can run alongside the DDS-to-ZMQ body bridge in the same server process.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _zmq_available

if TYPE_CHECKING or _zmq_available:
    import zmq
else:
    zmq = None  # type: ignore[assignment]

from .g1_ah_devices import HeadHandDevice

HEADHAND_CMD_PORT = 6002
HEADHAND_STATE_PORT = 6003
HEADHAND_STATE_TOPIC = "unitree_g1_ah/headhand_state"
HEADHAND_CMD_TOPIC = "unitree_g1_ah/headhand_cmd"


def encode_state(ticks: Mapping[str, int], torque: bool, error: str | None = None) -> bytes:
    """Serialize a head/hand state message to JSON bytes."""
    payload = {
        "topic": HEADHAND_STATE_TOPIC,
        "data": {"t": time.time(), "ticks": dict(ticks), "torque": torque, "error": error},
    }
    return json.dumps(payload).encode("utf-8")


def decode_state(payload: bytes) -> dict[str, Any]:
    """Deserialize a head/hand state message from JSON bytes."""
    msg = json.loads(payload.decode("utf-8"))
    return msg.get("data", {})


def encode_cmd(goal_ticks: Mapping[str, int] | None = None, torque: bool | None = None) -> bytes:
    """Serialize a head/hand command message to JSON bytes."""
    payload = {
        "topic": HEADHAND_CMD_TOPIC,
        "data": {"goal_ticks": dict(goal_ticks) if goal_ticks is not None else None, "torque": torque},
    }
    return json.dumps(payload).encode("utf-8")


def decode_cmd(payload: bytes) -> dict[str, Any]:
    """Deserialize a head/hand command message from JSON bytes."""
    msg = json.loads(payload.decode("utf-8"))
    return msg.get("data", {})


class HeadHandZmqClient:
    """ZMQ client for reading head/hand state and sending goal-tick commands."""

    def __init__(
        self,
        ip: str,
        state_port: int = HEADHAND_STATE_PORT,
        cmd_port: int = HEADHAND_CMD_PORT,
        *,
        context: zmq.Context | None = None,
    ) -> None:
        self.ip = ip
        self.state_port = state_port
        self.cmd_port = cmd_port
        self._owns_context = context is None
        self._context = context
        self._state_sock: zmq.Socket | None = None
        self._cmd_sock: zmq.Socket | None = None
        self._last_state: dict[str, Any] | None = None
        self.last_state_time: float | None = None

    def connect(self) -> None:
        if self._context is None:
            self._context = zmq.Context()

        state_sock = self._context.socket(zmq.SUB)
        state_sock.setsockopt(zmq.CONFLATE, 1)
        state_sock.setsockopt(zmq.RCVTIMEO, 50)
        state_sock.setsockopt_string(zmq.SUBSCRIBE, "")
        state_sock.connect(f"tcp://{self.ip}:{self.state_port}")
        self._state_sock = state_sock

        cmd_sock = self._context.socket(zmq.PUSH)
        cmd_sock.setsockopt(zmq.CONFLATE, 1)
        cmd_sock.setsockopt(zmq.LINGER, 0)
        cmd_sock.connect(f"tcp://{self.ip}:{self.cmd_port}")
        self._cmd_sock = cmd_sock

    def read_latest(self) -> dict[str, Any] | None:
        if self._state_sock is None:
            raise RuntimeError("Call connect() before read_latest().")

        latest_payload: bytes | None = None
        while True:
            try:
                latest_payload = self._state_sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

        if latest_payload is not None:
            self._last_state = decode_state(latest_payload)
            self.last_state_time = time.time()

        return self._last_state

    @property
    def age_s(self) -> float | None:
        if self.last_state_time is None:
            return None
        return time.time() - self.last_state_time

    def send(self, goal_ticks: Mapping[str, int] | None = None, torque: bool | None = None) -> None:
        if self._cmd_sock is None:
            raise RuntimeError("Call connect() before send().")
        payload = encode_cmd(goal_ticks=goal_ticks, torque=torque)
        with contextlib.suppress(zmq.Again):
            self._cmd_sock.send(payload, zmq.NOBLOCK)

    def disconnect(self) -> None:
        if self._state_sock is not None:
            self._state_sock.close()
            self._state_sock = None
        if self._cmd_sock is not None:
            self._cmd_sock.close()
            self._cmd_sock = None
        if self._owns_context and self._context is not None:
            self._context.term()
            self._context = None

    @property
    def is_connected(self) -> bool:
        return self._state_sock is not None and self._cmd_sock is not None


class HeadHandServer:
    """Runs a PULL/PUB ZMQ bridge that reads/writes ticks on a `HeadHandDevice`."""

    def __init__(
        self,
        device: HeadHandDevice,
        *,
        state_port: int = HEADHAND_STATE_PORT,
        cmd_port: int = HEADHAND_CMD_PORT,
        rate_hz: float = 50.0,
        context: zmq.Context | None = None,
        bind_address: str = "*",
    ) -> None:
        self.device = device
        self.state_port = state_port
        self.cmd_port = cmd_port
        self.rate_hz = rate_hz
        self.bind_address = bind_address
        self._owns_context = context is None
        self._context = context
        self._state_sock: zmq.Socket | None = None
        self._cmd_sock: zmq.Socket | None = None
        self._torque_on = False

    def bind(self) -> tuple[int, int]:
        if self._context is None:
            self._context = zmq.Context()

        state_sock = self._context.socket(zmq.PUB)
        if self.state_port == 0:
            self.state_port = state_sock.bind_to_random_port(f"tcp://{self.bind_address}")
        else:
            state_sock.bind(f"tcp://{self.bind_address}:{self.state_port}")
        self._state_sock = state_sock

        cmd_sock = self._context.socket(zmq.PULL)
        if self.cmd_port == 0:
            self.cmd_port = cmd_sock.bind_to_random_port(f"tcp://{self.bind_address}")
        else:
            cmd_sock.bind(f"tcp://{self.bind_address}:{self.cmd_port}")
        self._cmd_sock = cmd_sock

        return self.state_port, self.cmd_port

    def _drain_cmds(self) -> None:
        if self._cmd_sock is None:
            return
        while True:
            try:
                payload = self._cmd_sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            cmd = decode_cmd(payload)
            if cmd.get("torque") is not None:
                self._torque_on = bool(cmd["torque"])
                self.device.set_torque(self._torque_on)
            goal_ticks = cmd.get("goal_ticks")
            if goal_ticks:
                self.device.write_ticks(goal_ticks)

    def run(self, shutdown_event: threading.Event) -> None:
        if self._state_sock is None or self._cmd_sock is None:
            self.bind()

        period = 1.0 / self.rate_hz
        first_iteration = True
        while not shutdown_event.is_set():
            t0 = time.time()
            error: str | None = None
            ticks: dict[str, int] = {}
            try:
                self._drain_cmds()
                ticks = self.device.read_ticks()
                if first_iteration:
                    self.device.write_ticks(ticks)
                    self.device.set_torque(True)
                    self._torque_on = True
                    first_iteration = False
            except Exception as e:
                error = str(e)

            payload = encode_state(ticks, self._torque_on, error=error)
            with contextlib.suppress(zmq.Again):
                self._state_sock.send(payload, zmq.NOBLOCK)

            sleep = period - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def stop(self) -> None:
        if self._state_sock is not None:
            self._state_sock.close()
            self._state_sock = None
        if self._cmd_sock is not None:
            self._cmd_sock.close()
            self._cmd_sock = None
        if self._owns_context and self._context is not None:
            self._context.term()
            self._context = None
