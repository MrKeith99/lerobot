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

"""Hardware-light keyboard teleoperator for the UnitreeG1Ah robot.

Subclasses `UnitreeG1AhGamepadTeleop`, reusing its target-state stepping logic
(`_step_head`, `_step_hands`, `_remote_axes`, `get_action`, `get_teleop_events`)
unmodified. Only `connect()` is overridden, to create a `UnitreeG1AhKeyboardInput`
instead of a `UnitreeG1AhGamepadInput`.
"""

from __future__ import annotations

from ..unitree_g1_ah_gamepad.unitree_g1_ah_gamepad import UnitreeG1AhGamepadTeleop
from .config_unitree_g1_ah_keyboard import UnitreeG1AhKeyboardTeleopConfig
from .keyboard_input import UnitreeG1AhKeyboardInput


class UnitreeG1AhKeyboardTeleop(UnitreeG1AhGamepadTeleop):
    """Keyboard teleoperator emitting the full UnitreeG1Ah teleop action space (45 keys)."""

    config_class = UnitreeG1AhKeyboardTeleopConfig
    name = "unitree_g1_ah_keyboard"

    def connect(self, calibrate: bool = True) -> None:
        self.gamepad = UnitreeG1AhKeyboardInput(self.config)
        self.gamepad.start()
        self._last_t = None
        print("UnitreeG1Ah keyboard controls:")
        print("  Arrow keys: head pan (left/right) / tilt (up/down)")
        print("  q / e: hold to close left / right hand")
        print("  w/s, a/d: remote.ly / remote.lx (left stick)")
        print("  i/k, j/l: remote.ry / remote.rx (right stick)")
        print("  y / n / r: end episode success / failure / rerecord")
        print("  space: hold for intervention")
        print("  Esc: stop")
