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

"""DDS driver for the Unitree Dex3-1 hand and Dex1-1 gripper, one command/state topic pair per side.

Real hardware and the MuJoCo sim differ in three ways, all handled here:
- Dex3 real right hand orders its motors thumb, index, middle; the sim model orders both hands
  thumb, middle, index (the `DEX3` key order).
- Dex1 real hardware speaks `unitree_go` `MotorCmds_`/`MotorStates_` on `rt/dex1/*`, one motor per
  side; the sim serves it on the Dex3 `unitree_hg` `HandCmd_`/`HandState_` topics, one motor per finger.
- Dex1 real positions are motor units, converted linearly from the sim's finger travel in m.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _unitree_sdk_available

from .end_effectors import DEX1, HAND_SIDES, HandSpec

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__HandCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
else:
    unitree_go_msg_dds__MotorCmd_ = unitree_hg_msg_dds__HandCmd_ = None  # noqa: N816
    MotorCmds_ = MotorStates_ = HandCmd_ = HandState_ = None

# Real Dex1-1 gripper motor positions for fully open and fully closed. Verify on hardware.
DEX1_REAL_OPEN_Q = 5.4
DEX1_REAL_CLOSED_Q = 0.0

# (kp, kd) by (end effector, is_simulation).
DEX_GAINS: dict[tuple[str, bool], tuple[float, float]] = {
    ("dex3", False): (1.5, 0.2),
    ("dex3", True): (1.5, 0.2),
    ("dex1", False): (5.0, 0.05),
    ("dex1", True): (1000.0, 25.0),
}


def dex_topics(end_effector: str, is_simulation: bool, side: str) -> tuple[str, str]:
    """(command topic, state topic) of one side."""
    family = "dex1" if end_effector == "dex1" and not is_simulation else "dex3"
    return f"rt/{family}/{side}/cmd", f"rt/{family}/{side}/state"


def _motor_slots(spec: HandSpec, is_simulation: bool, side: str) -> tuple[tuple[int, ...], ...]:
    """For each spec joint of one side, the message motor slots it drives (the first is read back)."""
    names = spec.motor_names[side]
    if spec.name == "dex1":
        return ((0, 1),) if is_simulation else ((0,),)
    if spec.name == "dex3" and side == "right" and not is_simulation:
        real_order = ("Thumb0", "Thumb1", "Thumb2", "Index0", "Index1", "Middle0", "Middle1")
        return tuple((real_order.index(name.removeprefix("kRightHand")),) for name in names)
    return tuple((i,) for i in range(len(names)))


def _dex3_mode(motor_id: int) -> int:
    """Dex3 `RIS_Mode` byte: 4-bit motor id, status 0x01 (enabled), no timeout."""
    return (motor_id & 0x0F) | (0x01 << 4)


class DexHandDriver:
    """Commands and reads a Dex3 or Dex1 over DDS (or the ZMQ stand-in), in `spec` units."""

    def __init__(self, spec: HandSpec, is_simulation: bool, publisher_cls, subscriber_cls) -> None:
        self.spec = spec
        self.is_simulation = is_simulation
        self._publisher_cls = publisher_cls
        self._subscriber_cls = subscriber_cls
        self._real_dex1 = spec.name == "dex1" and not is_simulation
        self.kp, self.kd = DEX_GAINS[(spec.name, is_simulation)]
        self._slots = {side: _motor_slots(spec, is_simulation, side) for side in HAND_SIDES}
        self._lock = threading.Lock()
        self._q: dict[str, float] = {}
        self._publishers: dict[str, Any] = {}
        self._subscribers: dict[str, Any] = {}
        self._cmds: dict[str, Any] = {}

    def _message_types(self):
        """(command type, state type, command factory) of this hand's topics."""
        if self._real_dex1:
            return MotorCmds_, MotorStates_, lambda: MotorCmds_(cmds=[unitree_go_msg_dds__MotorCmd_()])
        return HandCmd_, HandState_, unitree_hg_msg_dds__HandCmd_

    def _to_motor(self, q: float) -> float:
        if not self._real_dex1:
            return q
        (open_m,), (closed_m,) = DEX1.open_q["left"], DEX1.closed_q["left"]
        fraction = (q - closed_m) / (open_m - closed_m)
        return DEX1_REAL_CLOSED_Q + fraction * (DEX1_REAL_OPEN_Q - DEX1_REAL_CLOSED_Q)

    def _from_motor(self, q: float) -> float:
        if not self._real_dex1:
            return q
        (open_m,), (closed_m,) = DEX1.open_q["left"], DEX1.closed_q["left"]
        fraction = (q - DEX1_REAL_CLOSED_Q) / (DEX1_REAL_OPEN_Q - DEX1_REAL_CLOSED_Q)
        return closed_m + fraction * (open_m - closed_m)

    @staticmethod
    def _motor_list(msg: Any, attr: str, fallback: str) -> list:
        return getattr(msg, attr) if hasattr(msg, attr) else getattr(msg, fallback)

    def _on_state(self, side: str, msg: Any) -> None:
        states = self._motor_list(msg, "motor_state", "states")
        keys = self.spec.joint_keys(side)
        with self._lock:
            for key, slots in zip(keys, self._slots[side], strict=True):
                if slots[0] < len(states):
                    self._q[key] = self._from_motor(float(states[slots[0]].q))

    def connect(self, timeout_s: float = 5.0) -> None:
        """Create both sides' channels and wait up to `timeout_s` for every joint's first state."""
        cmd_type, state_type, make_cmd = self._message_types()
        for side in HAND_SIDES:
            cmd_topic, state_topic = dex_topics(self.spec.name, self.is_simulation, side)
            publisher = self._publisher_cls(cmd_topic, cmd_type)
            publisher.Init()
            subscriber = self._subscriber_cls(state_topic, state_type)
            subscriber.Init(functools.partial(self._on_state, side), 1)
            self._publishers[side] = publisher
            self._subscribers[side] = subscriber
            self._cmds[side] = make_cmd()

        deadline = time.time() + timeout_s
        while len(self.read()) < len(self.spec.joint_keys()):
            if time.time() > deadline:
                missing = sorted(set(self.spec.joint_keys()) - set(self.read()))
                raise TimeoutError(f"Timed out waiting for {self.spec.name} state: {missing}")
            time.sleep(0.01)

    def read(self) -> dict[str, float]:
        """Latest joint positions, keyed as `spec.joint_keys()`; empty until the first state arrives."""
        with self._lock:
            return dict(self._q)

    def write(self, targets: Mapping[str, float]) -> None:
        """Publish the targets of each side that has at least one key in `targets`."""
        for side in HAND_SIDES:
            keys = self.spec.joint_keys(side)
            if not any(key in targets for key in keys):
                continue
            cmd = self._cmds[side]
            motors = self._motor_list(cmd, "motor_cmd", "cmds")
            for key, slots in zip(keys, self._slots[side], strict=True):
                if key not in targets:
                    continue
                for slot in slots:
                    motor = motors[slot]
                    motor.q = self._to_motor(float(targets[key]))
                    motor.dq = 0.0
                    motor.tau = 0.0
                    motor.kp = self.kp
                    motor.kd = self.kd
                    if self.spec.name == "dex3":
                        motor.mode = _dex3_mode(slot)
            self._publishers[side].Write(cmd)

    def disconnect(self) -> None:
        for subscriber in self._subscribers.values():
            close = getattr(subscriber, "Close", None)
            if close is not None:
                close()
        self._publishers.clear()
        self._subscribers.clear()
