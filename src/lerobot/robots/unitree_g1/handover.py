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

"""Relay side of taking low-level control from Unitree's locomotion service and handing it back.

Strategies (`run_g1_server.py --handover`):
- "release": release Unitree's mode at start; no handover, a client exit ends in damping (default).
- "user-ctrl": `LocoClient.SwitchToUserCtrl()` at start, `SwitchToInternalCtrl(WALKRUN)` to hand back. Needs a
  firmware with loco API 7110/7111 (SDK July 2026).
- "select-mode": release at start; to hand back, `SelectMode("ai")`, wait for the loco service, then
  `SetFsmId(fsm)` (500 = walk/run).

Both handovers are unverified on our robot: test them on the gantry first.
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable

HANDOVER_STRATEGIES = ("release", "user-ctrl", "select-mode")
DEFAULT_HANDOVER_FSM = 500
# Unitree loco FSM ids (`LocoClient`): 0 zero torque, 1 damping.
FSM_DAMP = 1


class HandoverState(str, enum.Enum):
    IDLE = "idle"
    HOLDING = "holding"  # client asked; the relay holds the client's stand command while switching
    DONE = "done"  # Unitree's controller owns the robot; the relay publishes nothing more
    FAILED = "failed"  # switching failed; the relay keeps holding until the client damps or the relay stops
    UNSUPPORTED = "unsupported"  # the relay runs without a handover strategy
    CANCELLED = "cancelled"  # the client sent a normal command (e.g. damping) during the handover


class Handover:
    """Shared between the relay's command loop (requests, publishing gate) and state loop (reports `state`)."""

    def __init__(self, give_back: Callable[[Callable[[], None]], str | None] | None) -> None:
        # give_back(stop_publishing) switches control to Unitree; it calls stop_publishing() right after the
        # switch is accepted, and returns None on success or an error message.
        self._give_back = give_back
        self.state = HandoverState.IDLE
        self.error: str | None = None
        self.publishing_stopped = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def supported(self) -> bool:
        return self._give_back is not None

    @property
    def active(self) -> bool:
        """True while the relay holds the robot for a handover (watchdog suspended)."""
        return self.state in (HandoverState.HOLDING, HandoverState.FAILED)

    def request(self) -> None:
        if not self.supported:
            self.state = HandoverState.UNSUPPORTED
            return
        if self.state not in (HandoverState.IDLE, HandoverState.CANCELLED):
            return
        self.state = HandoverState.HOLDING
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        if self.state in (HandoverState.HOLDING, HandoverState.FAILED):
            self.state = HandoverState.CANCELLED

    def _run(self) -> None:
        try:
            error = self._give_back(self.publishing_stopped.set)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
        if self.state == HandoverState.CANCELLED:
            if error is None or self.publishing_stopped.is_set():
                print("handover: finished after the client cancelled it; check who controls the robot")
            return
        if error is None:
            self.publishing_stopped.set()
            self.state = HandoverState.DONE
        else:
            self.publishing_stopped.clear()
            self.error = error
            self.state = HandoverState.FAILED
        print(f"handover: {self.state.value}" + (f" ({error})" if error else ""))


def relay_state(forwarding: bool, handover: Handover) -> str:
    """What the relay does with the robot, reported in every lowstate as `relay_state`."""
    if handover.state == HandoverState.DONE:
        return "handed_over"
    if handover.active:
        return "handover"
    return "forwarding" if forwarding else "damping"


def check_damping(fsm_id: int | None, assume_damping: bool) -> str | None:
    """None if the robot may be taken over, else why not. `fsm_id` is Unitree's loco FSM (None: no answer)."""
    if fsm_id == FSM_DAMP:
        return None
    if fsm_id is None:
        if assume_damping:
            return None
        return (
            "Unitree's loco service does not answer, so damping cannot be confirmed (e.g. a previous relay "
            "released it). Check the robot is in damping, then pass --assume-damping."
        )
    return (
        f"the robot is in loco FSM {fsm_id}, not damping ({FSM_DAMP}). Put it in damping with the Unitree remote, "
        "then start the relay again."
    )


def wait_for(predicate: Callable[[], bool], timeout_s: float, period_s: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(period_s)
    return predicate()


def make_strategy(name: str, fsm: int = DEFAULT_HANDOVER_FSM, assume_damping: bool = False):
    """(take_control, give_back) for a strategy name; needs the Unitree SDK and an initialized DDS factory.
    take_control() raises unless the robot is confirmed in damping (see `check_damping`)."""
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.g1.loco.g1_loco_api import InternalFsmMode
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    loco = LocoClient()
    loco.SetTimeout(2.0)
    loco.Init()

    def fsm_id() -> int | None:
        try:
            code, data = loco.GetFsmId()
        except Exception:  # noqa: BLE001
            return None
        return data if code == 0 else None

    def require_damping() -> None:
        current = fsm_id()
        reason = check_damping(current, assume_damping)
        if reason is not None:
            raise RuntimeError(f"refusing to take control: {reason}")
        print(f"damping confirmed (loco fsm_id={current})" if current is not None else "damping assumed")

    def release() -> None:
        require_damping()
        status, result = msc.CheckMode()
        while result is not None and "name" in result and result["name"]:
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(1.0)

    if name == "release":
        return release, None

    if name == "user-ctrl":

        def take() -> None:
            require_damping()
            code = loco.SwitchToUserCtrl()
            if code != 0:
                raise RuntimeError(
                    f"SwitchToUserCtrl failed (code {code}); the firmware may lack loco API 7110. "
                    "Use --handover release or select-mode."
                )

        def give_back(stop_publishing: Callable[[], None]) -> str | None:
            code = loco.SwitchToInternalCtrl(InternalFsmMode.WALKRUN)
            if code != 0:
                return f"SwitchToInternalCtrl(WALKRUN) failed (code {code})"
            stop_publishing()
            print(f"handover: Unitree controller active, fsm_id={fsm_id()}")
            return None

        return take, give_back

    if name == "select-mode":

        def give_back(stop_publishing: Callable[[], None]) -> str | None:
            status, _ = msc.SelectMode("ai")
            if status != 0:
                return f"SelectMode('ai') failed (code {status})"
            if not wait_for(lambda: fsm_id() is not None, timeout_s=10.0):
                return "the loco service did not answer within 10 s after SelectMode('ai')"
            print(f"handover: loco service up, fsm_id={fsm_id()}; requesting fsm {fsm}")
            code = loco.SetFsmId(fsm)
            if code != 0:
                return f"SetFsmId({fsm}) failed (code {code})"
            stop_publishing()
            if not wait_for(lambda: fsm_id() == fsm, timeout_s=3.0):
                print(f"handover: warning, fsm_id={fsm_id()} after SetFsmId({fsm})")
            return None

        return release, give_back

    raise ValueError(f"handover strategy must be one of {HANDOVER_STRATEGIES}, got {name!r}")
