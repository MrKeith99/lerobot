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

import base64
import json
import threading
from collections.abc import Callable
from typing import Any

import zmq

from .config_unitree_g1 import UnitreeG1Config

# Module-level ZMQ state mirrors the Unitree SDK's global ChannelFactory Singleton.
# Only one robot connection per process is supported.
_ctx: zmq.Context | None = None
_robot_ip: str | None = None

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001

# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"

# One ZMQ port per bridged DDS topic (6002/6003 are the head/hand bridge). Commands are PUSH/PULL,
# states PUB/SUB, both conflated to the latest message, so each topic needs its own socket.
TOPIC_PORTS: dict[str, int] = {
    kTopicLowCommand_Debug: LOWCMD_PORT,
    kTopicLowState: LOWSTATE_PORT,
    "rt/dex3/left/cmd": 6010,
    "rt/dex3/right/cmd": 6011,
    "rt/dex3/left/state": 6012,
    "rt/dex3/right/state": 6013,
    "rt/dex1/left/cmd": 6014,
    "rt/dex1/right/cmd": 6015,
    "rt/dex1/left/state": 6016,
    "rt/dex1/right/state": 6017,
}


def topic_port(topic: str) -> int:
    """ZMQ port of a bridged DDS topic."""
    if topic not in TOPIC_PORTS:
        raise ValueError(f"Topic {topic!r} is not bridged; bridged topics: {list(TOPIC_PORTS)}")
    return TOPIC_PORTS[topic]


class LowStateMsg:
    """
    Wrapper class that mimics the Unitree SDK LowState_ message structure.

    Reconstructs the message from deserialized JSON data to maintain
    compatibility with existing code that expects SDK message objects.
    """

    class MotorState:
        """Motor state data for a single joint."""

        def __init__(self, data: dict[str, Any]) -> None:
            self.q: float = data.get("q", 0.0)
            self.dq: float = data.get("dq", 0.0)
            self.tau_est: float = data.get("tau_est", 0.0)
            self.temperature: float = data.get("temperature", 0.0)

    class IMUState:
        """IMU sensor data."""

        def __init__(self, data: dict[str, Any]) -> None:
            self.quaternion: list[float] = data.get("quaternion", [1.0, 0.0, 0.0, 0.0])
            self.gyroscope: list[float] = data.get("gyroscope", [0.0, 0.0, 0.0])
            self.accelerometer: list[float] = data.get("accelerometer", [0.0, 0.0, 0.0])
            self.rpy: list[float] = data.get("rpy", [0.0, 0.0, 0.0])
            self.temperature: float = data.get("temperature", 0.0)

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize from deserialized JSON data."""
        self.motor_state = [self.MotorState(m) for m in data.get("motor_state", [])]
        self.imu_state = self.IMUState(data.get("imu_state", {}))
        # Decode base64-encoded wireless_remote bytes
        wireless_b64 = data.get("wireless_remote", "")
        self.wireless_remote: bytes = base64.b64decode(wireless_b64) if wireless_b64 else b""
        self.mode_machine: int = data.get("mode_machine", 0)


class MotorStatesMsg:
    """Mimics a hand state message: `HandState_.motor_state` (Dex3) or `MotorStates_.states` (Dex1)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.motor_state = [LowStateMsg.MotorState(m) for m in data.get("motor_state", [])]
        self.states = self.motor_state


def lowcmd_to_dict(topic: str, msg: Any) -> dict[str, Any]:
    """Convert a motor command message (LowCmd_, HandCmd_ or MotorCmds_) to a JSON-serializable dictionary."""
    motors = msg.motor_cmd if hasattr(msg, "motor_cmd") else msg.cmds
    data: dict[str, Any] = {
        "motor_cmd": [
            {
                "mode": int(m.mode),
                "q": float(m.q),
                "dq": float(m.dq),
                "kp": float(m.kp),
                "kd": float(m.kd),
                "tau": float(m.tau),
            }
            for m in motors
        ]
    }
    if hasattr(msg, "mode_pr"):
        data["mode_pr"] = int(msg.mode_pr)
        data["mode_machine"] = int(msg.mode_machine)
    return {"topic": topic, "data": data}


def ChannelFactoryInitialize(domain_id: int = 0, config: Any = None) -> None:  # noqa: N802
    """
    Initialize ZMQ communication with the robot server bridge.

    This function mimics the Unitree SDK's ChannelFactoryInitialize but uses
    ZMQ sockets to connect to the robot server bridge instead of DDS.

    Args:
        domain_id: Ignored (for API compatibility with Unitree SDK)
        config: UnitreeG1Config instance with robot_ip
    """
    global _ctx, _robot_ip

    # read socket config
    if config is None:
        config = UnitreeG1Config()
    _robot_ip = config.robot_ip
    _ctx = zmq.Context.instance()


def _socket(kind: int, topic: str) -> zmq.Socket:
    if _ctx is None:
        raise RuntimeError("ChannelFactoryInitialize must be called first")
    sock = _ctx.socket(kind)
    sock.setsockopt(zmq.CONFLATE, 1)  # keep only last message
    sock.connect(f"tcp://{_robot_ip}:{topic_port(topic)}")
    return sock


class ChannelPublisher:
    """ZMQ-based publisher that sends commands to the robot server."""

    def __init__(self, topic: str, msg_type: type) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self._sock: zmq.Socket | None = None

    def Init(self) -> None:  # noqa: N802
        """Connect the command socket of this topic."""
        self._sock = _socket(zmq.PUSH, self.topic)

    def Write(self, msg: Any) -> None:  # noqa: N802
        """Serialize and send a command message to the robot."""
        if self._sock is None:
            raise RuntimeError("Init must be called first")

        payload = json.dumps(lowcmd_to_dict(self.topic, msg)).encode("utf-8")
        self._sock.send(payload)


class ChannelSubscriber:
    """ZMQ-based subscriber that receives state from the robot server.

    Like the SDK, it is either read with `Read()` or, when `Init` gets a handler, delivers each
    message to the handler from a background thread.
    """

    def __init__(self, topic: str, msg_type: type) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self._sock: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def Init(self, handler: Callable[[Any], None] | None = None, queueLen: int = 0) -> None:  # noqa: N802, N803
        """Connect the state socket of this topic, and start delivering to `handler` if given."""
        self._sock = _socket(zmq.SUB, self.topic)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        if handler is not None:
            self._sock.setsockopt(zmq.RCVTIMEO, 100)
            self._thread = threading.Thread(target=self._deliver, args=(handler,), daemon=True)
            self._thread.start()

    def _decode(self, payload: bytes) -> LowStateMsg | MotorStatesMsg:
        data = json.loads(payload.decode("utf-8")).get("data", {})
        return LowStateMsg(data) if self.topic == kTopicLowState else MotorStatesMsg(data)

    def _deliver(self, handler: Callable[[Any], None]) -> None:
        while not self._stop.is_set():
            try:
                payload = self._sock.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:  # socket closed or context terminated
                break
            handler(self._decode(payload))

    def Read(self) -> LowStateMsg | MotorStatesMsg:  # noqa: N802
        """Receive and deserialize a state message from the robot."""
        if self._sock is None:
            raise RuntimeError("Init must be called first")
        return self._decode(self._sock.recv())

    def Close(self) -> None:  # noqa: N802
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
