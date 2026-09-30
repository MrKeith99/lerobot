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

"""Hardware-light keyboard teleoperator for the Unitree G1.

Subclasses `UnitreeG1GamepadTeleop`, reusing its target-state stepping logic
(`_step_head`, `_step_hands`, `_remote_axes`, `get_action`, `get_teleop_events`)
unmodified. Only `connect()` is overridden, to create a `UnitreeG1KeyboardInput`
instead of a `UnitreeG1GamepadInput`.
"""

from __future__ import annotations

from ..unitree_g1_gamepad.unitree_g1_gamepad import UnitreeG1GamepadTeleop
from .config_unitree_g1_keyboard import UnitreeG1KeyboardTeleopConfig
from .keyboard_input import UnitreeG1KeyboardInput


class UnitreeG1KeyboardTeleop(UnitreeG1GamepadTeleop):
    """Keyboard teleoperator emitting the full Unitree G1 teleop action space (`TELEOP_ACTION_KEYS`).

    t / g stand in for the gamepad's L2 / R2: they raise / lower `kBaseHeight.cmd` and set the
    locomotion controller's waist raise/lower `REMOTE_BUTTONS` slots.
    """

    config_class = UnitreeG1KeyboardTeleopConfig
    name = "unitree_g1_keyboard"

    def connect(self, calibrate: bool = True) -> None:
        self.gamepad = UnitreeG1KeyboardInput(self.config)
        self.gamepad.start()
        self._last_t = None
        print("Unitree G1 keyboard controls:")
        print("  Arrow keys: head pan (left/right) / tilt (up/down)")
        print("  q / e: hold to close left / right hand")
        print("  w/s, a/d: forward / sideways (kNavVx / kNavVy, left stick)")
        print("  i/k, j/l: remote.ry / remote.rx (right stick)")
        print("  t / g: hold to raise / lower base height (GrootLocomotionController)")
        print("  y / n / r: end episode success / failure / rerecord")
        print("  space: hold for intervention")
        print("  Esc: stop")
