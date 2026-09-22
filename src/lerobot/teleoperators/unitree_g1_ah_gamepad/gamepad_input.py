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

import argparse
import logging
import time
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _pygame_available

from ..gamepad.gamepad_utils import GamepadController
from ..utils import TeleopEvents
from .config_unitree_g1_ah_gamepad import GamepadLayout

if TYPE_CHECKING or _pygame_available:
    import pygame
else:
    pygame = None  # type: ignore[assignment]


class UnitreeG1AhGamepadInput(GamepadController):
    """Gamepad input reader for UnitreeG1Ah teleop: raw axes/buttons/hat plus episode events.

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
        layout = self.layout
        if layout.dpad_up is not None:
            return _dpad_from_buttons(self, layout)
        try:
            if layout.hat is not None and self.joystick.get_numhats() > 0:
                return self.joystick.get_hat(layout.hat)
            return (0, 0)
        except pygame.error:
            return (0, 0)

    @property
    def is_running(self) -> bool:
        return self.running


def _dpad_from_buttons(gamepad: UnitreeG1AhGamepadInput, layout: GamepadLayout) -> tuple[int, int]:
    right = int(gamepad.button(layout.dpad_right))
    left = int(gamepad.button(layout.dpad_left))
    up = int(gamepad.button(layout.dpad_up))
    down = int(gamepad.button(layout.dpad_down))
    return (right - left, up - down)


def probe(seconds: float = 20.0) -> None:
    """Print live axis/button/hat activity for the first connected gamepad.

    Use this to discover a pad's index layout for a `GamepadLayout` preset:
    `python -m lerobot.teleoperators.unitree_g1_ah_gamepad.gamepad_input --seconds 30`.
    """
    if not _pygame_available:
        print("pygame is not installed. Install it with: pip install 'lerobot[gamepad]'")
        return

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No gamepad detected. Please connect a gamepad and try again.")
        pygame.quit()
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick: {joystick.get_name()}")
    print(f"axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}")
    print("Move sticks/triggers and press buttons/D-pad to see their indices. Ctrl+C to stop early.")

    last_hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
    start = time.monotonic()
    try:
        while time.monotonic() - start < seconds:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"button {event.button} down")
            for axis in range(joystick.get_numaxes()):
                value = joystick.get_axis(axis)
                if abs(value) > 0.5:
                    print(f"axis {axis} = {value:.2f}")
            for hat_index in range(joystick.get_numhats()):
                value = joystick.get_hat(hat_index)
                if value != last_hats[hat_index]:
                    print(f"hat {hat_index} = {value}")
                    last_hats[hat_index] = value
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        joystick.quit()
        pygame.joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe a connected gamepad's axis/button/hat indices.")
    parser.add_argument("--seconds", type=float, default=20.0, help="How long to poll for (default: 20).")
    args = parser.parse_args()
    probe(args.seconds)
