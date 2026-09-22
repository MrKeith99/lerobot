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

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class GamepadLayout:
    """pygame axis/button/hat indices. Defaults = Xbox-style controller under SDL2 on Linux."""

    left_x: int = 0
    left_y: int = 1
    right_x: int = 3
    right_y: int = 4
    button_a: int = 0
    button_b: int = 1
    button_x: int = 2
    button_y: int = 3
    button_lb: int = 4
    button_rb: int = 5
    button_back: int = 6
    button_start: int = 7
    hat: int = 0


@TeleoperatorConfig.register_subclass("g1ah_gamepad")
@dataclass
class G1AhGamepadTeleopConfig(TeleoperatorConfig):
    """Hardware-light gamepad teleoperator for the G1Ah robot.

    Emits every key in `TELEOP_ACTION_KEYS`: held body/hand poses, D-pad-driven
    head targets, an RB/LB-blended hand open/close, and the 4 `REMOTE_AXES`
    driven by the sticks (zeros when `emit_remote_axes` is False).
    """

    layout: GamepadLayout = field(default_factory=GamepadLayout)
    deadzone: float = 0.1
    head_speed_rad_s: float = 0.8
    hand_blend_per_s: float = 2.0
    emit_remote_axes: bool = True
    invert_tilt: bool = False
    initial_positions: dict[str, float] | None = None
