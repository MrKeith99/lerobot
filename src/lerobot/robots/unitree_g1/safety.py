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

"""Damping mode and the relay's lowcmd watchdog for the real G1.

Damping = kp 0, kd > 0, tau 0 on every body motor: the joints resist motion but hold no pose, so the robot
sinks slowly instead of dropping limp (zero torque) or freezing stiff (last command). The real robot is never
sent zero torque.
"""

from __future__ import annotations

from typing import Any

NUM_BODY_MOTORS = 29

# Per joint, SDK order: legs (hip pitch/roll/yaw, knee, ankle pitch/roll) x2, waist (yaw, roll, pitch),
# arms (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) x2.
DAMPING_KD: tuple[float, ...] = (
    *(2.0, 2.0, 2.0, 4.0, 2.0, 2.0) * 2,
    5.0,
    5.0,
    5.0,
    *(3.0, 3.0, 3.0, 3.0, 1.5, 1.5, 1.5) * 2,
)

# Relay: damping once no lowcmd arrived for this long; client: damping once lowstate is this old.
LOWCMD_TIMEOUT_S = 0.3
LOWSTATE_TIMEOUT_S = 0.2


def damping_motor_cmds(kd: tuple[float, ...] | list[float] = DAMPING_KD) -> list[dict[str, Any]]:
    """`motor_cmd` entries (the bridge's JSON form) putting every body motor in damping."""
    if len(kd) != NUM_BODY_MOTORS:
        raise ValueError(f"damping kd must have {NUM_BODY_MOTORS} entries, got {len(kd)}")
    return [{"mode": 1, "q": 0.0, "dq": 0.0, "kp": 0.0, "kd": float(k), "tau": 0.0} for k in kd]


class LowCmdWatchdog:
    """Relay-side gate between client lowcmds and the robot.

    Starts in damping. The first lowcmd of a client session starts forwarding; if no lowcmd arrives for
    `timeout_s`, it latches damping for that session (a stalled or reconnected-by-accident client cannot take
    the robot back). Only a new session, i.e. a freshly connected client, forwards again. A second session
    is ignored while one is active.
    """

    def __init__(self, timeout_s: float = LOWCMD_TIMEOUT_S) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        self.timeout_s = timeout_s
        self.session: str | None = None
        self.forwarding = False
        self._tripped: set[str | None] = set()
        self._last_cmd_t: float | None = None

    def on_command(self, session: str | None, now: float) -> bool:
        """A lowcmd from `session` arrived at `now`; True if it may be forwarded to the robot."""
        if session in self._tripped:
            return False
        if self.forwarding and session != self.session:
            return False  # a second client while one is active
        self.session = session
        self.forwarding = True
        self._last_cmd_t = now
        return True

    def check(self, now: float) -> bool:
        """True when the robot must be held in damping now; trips the active session on a timeout."""
        if self.forwarding and self._last_cmd_t is not None and now - self._last_cmd_t > self.timeout_s:
            self.forwarding = False
            self._tripped.add(self.session)
            return True
        return not self.forwarding

    def stale_s(self, now: float) -> float | None:
        return None if self._last_cmd_t is None else now - self._last_cmd_t
