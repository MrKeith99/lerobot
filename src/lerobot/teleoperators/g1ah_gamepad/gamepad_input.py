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

import logging
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _pygame_available

from ..gamepad.gamepad_utils import GamepadController
from ..utils import TeleopEvents
from .config_g1ah_gamepad import GamepadLayout

if TYPE_CHECKING or _pygame_available:
    import pygame
else:
    pygame = None  # type: ignore[assignment]


class G1AhGamepadInput(GamepadController):
    """Gamepad input reader for G1Ah teleop: raw axes/buttons/hat plus episode events.

    Button indices come from `GamepadLayout` instead of the hardcoded ones in
    `GamepadController.update()`. `intervention_flag` (read back through the inherited
    `should_intervene()`) is set while the Start button is held.
    """

    def __init__(self, layout: GamepadLayout, deadzone: float):
        super().__init__(deadzone=deadzone)
        self.layout = layout

    def update(self) -> None:
        if self.joystick is None:
            return
        try:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    self._on_button_down(event.button)
                elif event.type == pygame.JOYBUTTONUP:
                    self._on_button_up(event.button)
            self.intervention_flag = bool(self.joystick.get_button(self.layout.button_start))
        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")

    def _on_button_down(self, button: int) -> None:
        if button == self.layout.button_y:
            self.episode_end_status = TeleopEvents.SUCCESS
        elif button == self.layout.button_a:
            self.episode_end_status = TeleopEvents.FAILURE
        elif button == self.layout.button_x:
            self.episode_end_status = TeleopEvents.RERECORD_EPISODE

    def _on_button_up(self, button: int) -> None:
        if button in (self.layout.button_y, self.layout.button_a, self.layout.button_x):
            self.episode_end_status = None

    def axis(self, index: int) -> float:
        if self.joystick is None:
            return 0.0
        try:
            value = self.joystick.get_axis(index)
        except pygame.error:
            return 0.0
        return 0.0 if abs(value) < self.deadzone else value

    def button(self, index: int) -> bool:
        if self.joystick is None:
            return False
        try:
            return bool(self.joystick.get_button(index))
        except pygame.error:
            return False

    def hat(self) -> tuple[int, int]:
        if self.joystick is None:
            return (0, 0)
        try:
            if self.joystick.get_numhats() > 0:
                return self.joystick.get_hat(self.layout.hat)
            return (0, 0)
        except pygame.error:
            return (0, 0)

    @property
    def is_running(self) -> bool:
        return self.running
