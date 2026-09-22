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

"""In-process fake G1Ah head/hand server for ZMQ client tests. No hardware/SDK required."""

from __future__ import annotations

import contextlib
import threading
import time

import zmq

from lerobot.robots.g1ah.g1ah_devices import default_calibration
from lerobot.robots.g1ah.g1ah_joints import HEAD_HAND_MOTORS
from lerobot.robots.g1ah.g1ah_zmq import decode_cmd, encode_state


class MockHeadHandServer:
    """Fake head/hand bridge: publishes state and applies incoming goal ticks, no real bus."""

    def __init__(self, rate_hz: float = 100.0) -> None:
        self.rate_hz = rate_hz
        self.ticks: dict[str, int] = {
            name: default_calibration(name).homing_offset for name in HEAD_HAND_MOTORS
        }
        self.torque = False
        self.received: list[dict] = []

        self._context = zmq.Context()
        self._state_sock = self._context.socket(zmq.PUB)
        self.state_port = self._state_sock.bind_to_random_port("tcp://127.0.0.1")
        self._cmd_sock = self._context.socket(zmq.PULL)
        self.cmd_port = self._cmd_sock.bind_to_random_port("tcp://127.0.0.1")

        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        while not self._shutdown.is_set():
            t0 = time.time()
            while True:
                try:
                    payload = self._cmd_sock.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                cmd = decode_cmd(payload)
                self.received.append(cmd)
                if cmd.get("torque") is not None:
                    self.torque = bool(cmd["torque"])
                goal_ticks = cmd.get("goal_ticks")
                if goal_ticks:
                    self.ticks.update(goal_ticks)

            payload = encode_state(self.ticks, self.torque)
            with contextlib.suppress(zmq.Again):
                self._state_sock.send(payload, zmq.NOBLOCK)

            sleep = period - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._state_sock.close()
        self._cmd_sock.close()
        self._context.term()

    def __enter__(self) -> MockHeadHandServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
