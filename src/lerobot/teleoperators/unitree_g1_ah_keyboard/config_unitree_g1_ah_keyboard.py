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

from dataclasses import dataclass

from ..config import TeleoperatorConfig
from ..unitree_g1_ah_gamepad.config_unitree_g1_ah_gamepad import UnitreeG1AhGamepadTeleopConfig


@TeleoperatorConfig.register_subclass("unitree_g1_ah_keyboard")
@dataclass
class UnitreeG1AhKeyboardTeleopConfig(UnitreeG1AhGamepadTeleopConfig):
    """Hardware-light keyboard teleoperator for the UnitreeG1Ah robot.

    Reuses `UnitreeG1AhGamepadTeleop`'s target-state stepping logic (head, hand blend,
    remote axes) against a `pynput`-backed keyboard input instead of a joystick. Emits
    every key in `TELEOP_ACTION_KEYS`, same as the gamepad teleop.

    `layout` and `preset` are inherited from `UnitreeG1AhGamepadTeleopConfig` but unused
    here: the keyboard input has a fixed key mapping (arrows/WASD/IJKL/QE), not indices.

    `remote_axis_value` is the stick magnitude emitted (via `axis()`) while a
    WASD/IJKL key is held, in place of a gamepad's continuous analog stick.
    """

    remote_axis_value: float = 0.6
