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

"""Keyboard input reader for `UnitreeG1AhKeyboardTeleop`.

Mirrors the duck-typed interface `UnitreeG1AhGamepadInput` exposes to
`UnitreeG1AhGamepadTeleop` (`start`, `stop`, `update`, `axis`, `button`, `hat`,
`should_intervene`, `get_episode_end_status`, `is_running`) so the teleop's
target-state stepping logic (`_step_head`, `_step_hands`, `_remote_axes`) works
unmodified against a `pynput.keyboard.Listener` instead of a joystick.

Key mapping:
    Arrow keys: head pan/tilt, read through `hat()` -> (x, y). Left = pan left
        (x=-1, matching a gamepad D-pad's `Left arrow turns the head left`
        convention from `UnitreeG1AhGamepadTeleop._step_head`), Right = pan right
        (x=+1), Up = tilt up (y=+1), Down = tilt down (y=-1).
    q / e: held -> left / right hand closed, read through `button()` at the
        config layout's `button_lb` / `button_rb` indices (regardless of preset).
    w/s, a/d: `remote.ly` / `remote.lx`, read through `axis()` at the config
        layout's `left_y` / `left_x` indices, using pygame's sign convention
        (y axes point down: w -> negative, s -> positive).
    i/k, j/l: `remote.ry` / `remote.rx`, same convention at `right_y` / `right_x`.
    y / n / r: latch SUCCESS / FAILURE / RERECORD_EPISODE while held, cleared on release.
    space: held -> intervention.
    Esc: stop the listener (`is_running` becomes False).

Without a display, or when `pynput` cannot capture keys (Wayland, headless), `start()`
logs a warning and the input runs with no keys ever pressed, like `teleop_keyboard.py`.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.keyboard_input import pynput_can_capture

from ..utils import TeleopEvents
from .config_unitree_g1_ah_keyboard import UnitreeG1AhKeyboardTeleopConfig

if TYPE_CHECKING or _pynput_available:
    import pynput.keyboard as pynput_keyboard
else:
    pynput_keyboard = None  # type: ignore[assignment]

_ARROW_HAT: dict[str, tuple[int, int]] = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, 1),
    "down": (0, -1),
}
_EPISODE_KEYS: dict[str, TeleopEvents] = {
    "y": TeleopEvents.SUCCESS,
    "n": TeleopEvents.FAILURE,
    "r": TeleopEvents.RERECORD_EPISODE,
}


class UnitreeG1AhKeyboardInput:
    """`pynput`-backed keyboard input exposing the same interface as `UnitreeG1AhGamepadInput`."""

    def __init__(self, config: UnitreeG1AhKeyboardTeleopConfig):
        self.config = config
        self._pressed: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None
        self.running = True
        self.intervention_flag = False
        self.episode_end_status: TeleopEvents | None = None

    def start(self) -> None:
        if not (_pynput_available and pynput_can_capture()):
            logging.warning(
                "Keyboard teleoperation is unavailable in this environment. pynput can only "
                "capture key events on an X11 session (Linux), a Windows desktop, or macOS with "
                "Accessibility / Input Monitoring granted - not on Wayland or headless machines. "
                "This keyboard teleoperator will produce no actions; use an X11 session or a "
                "gamepad teleoperator instead."
            )
            return
        self._listener = pynput_keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self.running = False

    def update(self) -> None:
        pass

    def press(self, name: str) -> None:
        """Test/manual helper: mark `name` as held (see `_key_name` for the naming scheme)."""
        with self._lock:
            self._pressed.add(name)
        self._on_name_down(name)

    def release(self, name: str) -> None:
        """Test/manual helper: mark `name` as released."""
        with self._lock:
            self._pressed.discard(name)
        self._on_name_up(name)

    def _on_press(self, key) -> None:
        name = _key_name(key)
        if name is None:
            return
        self.press(name)

    def _on_release(self, key) -> None:
        name = _key_name(key)
        if name is None:
            return
        self.release(name)

    def _on_name_down(self, name: str) -> None:
        if name == "space":
            self.intervention_flag = True
        elif name in _EPISODE_KEYS:
            self.episode_end_status = _EPISODE_KEYS[name]

    def _on_name_up(self, name: str) -> None:
        if name == "space":
            self.intervention_flag = False
        elif name == "esc":
            self.running = False
        elif name in _EPISODE_KEYS and self.episode_end_status == _EPISODE_KEYS[name]:
            self.episode_end_status = None

    def hat(self) -> tuple[int, int]:
        with self._lock:
            pressed = set(self._pressed)
        x, y = 0, 0
        for direction, (dx, dy) in _ARROW_HAT.items():
            if direction in pressed:
                x += dx
                y += dy
        return (x, y)

    def button(self, index: int) -> bool:
        layout = self.config.layout
        with self._lock:
            pressed = set(self._pressed)
        if index == layout.button_lb:
            return "q" in pressed
        if index == layout.button_rb:
            return "e" in pressed
        return False

    def axis(self, index: int) -> float:
        layout = self.config.layout
        value = self.config.remote_axis_value
        with self._lock:
            pressed = set(self._pressed)
        if index == layout.left_x:
            return (value if "d" in pressed else 0.0) - (value if "a" in pressed else 0.0)
        if index == layout.left_y:
            return (value if "s" in pressed else 0.0) - (value if "w" in pressed else 0.0)
        if index == layout.right_x:
            return (value if "l" in pressed else 0.0) - (value if "j" in pressed else 0.0)
        if index == layout.right_y:
            return (value if "k" in pressed else 0.0) - (value if "i" in pressed else 0.0)
        return 0.0

    def should_intervene(self) -> bool:
        return self.intervention_flag

    def get_episode_end_status(self) -> TeleopEvents | None:
        return self.episode_end_status

    @property
    def is_running(self) -> bool:
        return self.running


def _key_name(key) -> str | None:
    """Normalize a `pynput` key event to the lowercase names this module matches on."""
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    special = {
        pynput_keyboard.Key.up: "up",
        pynput_keyboard.Key.down: "down",
        pynput_keyboard.Key.left: "left",
        pynput_keyboard.Key.right: "right",
        pynput_keyboard.Key.space: "space",
        pynput_keyboard.Key.esc: "esc",
    }
    return special.get(key)
