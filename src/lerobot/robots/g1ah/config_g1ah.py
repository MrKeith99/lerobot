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

"""Config for the G1Ah robot: Unitree G1 23dof body + Dynamixel head + AmazingHand hands."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

from ..config import RobotConfig
from .g1ah_joints import (
    DEFAULT_HAND_Q,
    DEFAULT_HEAD_Q,
    G1_23_INVALID_SDK_SLOTS,
    G1_23_LEG_SLOTS,
    REVISIONS,
    ROBOT_TYPE_BASE,
    ROBOT_TYPE_BY_REVISION,
)
from .g1ah_zmq import HEADHAND_CMD_PORT, HEADHAND_STATE_PORT


@RobotConfig.register_subclass(ROBOT_TYPE_BASE)
@dataclass
class G1AhConfig(UnitreeG1Config):
    revision: str = "rev_1_0"
    check_mode_machine: bool = True
    is_simulation: bool = False
    headhand_state_port: int = HEADHAND_STATE_PORT
    headhand_cmd_port: int = HEADHAND_CMD_PORT
    headhand_timeout_s: float = 5.0
    headhand_stale_warn_s: float = 0.5
    head_default_positions: list[float] = field(default_factory=lambda: list(DEFAULT_HEAD_Q))
    hand_default_positions: list[float] = field(
        default_factory=lambda: [*DEFAULT_HAND_Q["left"], *DEFAULT_HAND_Q["right"]]
    )
    freeze_legs: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.revision not in REVISIONS:
            raise ValueError(f"revision must be one of {REVISIONS}, got {self.revision!r}")
        if not (len(self.kp) == len(self.kd) == len(self.default_positions) == 29):
            raise ValueError("kp, kd and default_positions must all have length 29")
        if len(self.head_default_positions) != 2:
            raise ValueError("head_default_positions must have length 2")
        if len(self.hand_default_positions) != 16:
            raise ValueError("hand_default_positions must have length 16")

        for i in G1_23_INVALID_SDK_SLOTS:
            self.kp[i] = 0.0
            self.kd[i] = 0.0

        if self.freeze_legs and self.controller is None:
            for i in G1_23_LEG_SLOTS:
                self.kp[i] = 0.0
                self.kd[i] = 0.0

    @property
    def robot_type_name(self) -> str:
        return ROBOT_TYPE_BY_REVISION[self.revision]
