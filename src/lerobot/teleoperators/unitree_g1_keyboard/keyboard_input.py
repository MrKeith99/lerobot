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

"""Keyboard input reader for `UnitreeG1KeyboardTeleop`.

Mirrors the duck-typed interface `UnitreeG1GamepadInput` exposes to
`UnitreeG1GamepadTeleop` (`start`, `stop`, `update`, `axis`, `button`, `hat`,
`should_intervene`, `get_episode_end_status`, `is_running`) so the teleop's
target-state stepping logic (`_step_head`, `_step_hands`, `_remote_axes`) works
unmodified against a `pynput.keyboard.Listener` instead of a joystick.

Key mapping:
    Arrow keys: head pan/tilt, read through `hat()` -> (x, y). Left = pan left
        (x=-1, matching a gamepad D-pad's `Left arrow turns the head left`
        convention from `UnitreeG1GamepadTeleop._step_head`), Right = pan right
        (x=+1), Up = tilt up (y=+1), Down = tilt down (y=-1).
    q / e: held -> left / right hand closed, read through `button()` at the
        config layout's `button_lb` / `button_rb` indices (regardless of preset).
    w/s, a/d: `remote.ly` / `remote.lx`, read through `axis()` at the config
        layout's `left_y` / `left_x` indices, using pygame's sign convention
        (y axes point down: w -> negative, s -> positive).
    i/k, j/l: `remote.ry` / `remote.rx`, same convention at `right_y` / `right_x`.
    t / g: held -> raise / lower the base height, read through `axis()` at the config layout's
        `trigger_left` / `trigger_right` indices (the gamepad's L2 / R2).
    y / n / r: latch SUCCESS / FAILURE / RERECORD_EPISODE while held, cleared on release.
    space: held -> intervention.
    Esc: stop the listener (`is_running` becomes False).

Without a display, or when `pynput` cannot capture keys (Wayland, headless), `start()`
logs a warning and the input runs with no keys ever pressed, like `teleop_keyboard.py`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _pygame_available, _pynput_available
from lerobot.utils.keyboard_input import pynput_can_capture

from ..utils import TeleopEvents
from .config_unitree_g1_keyboard import UnitreeG1KeyboardTeleopConfig

pynput_keyboard = None
if TYPE_CHECKING or _pynput_available:
    try:
        import pynput.keyboard as pynput_keyboard
    except Exception as e:
        logging.info("Could not import pynput keyboard backend: %s", e)

if TYPE_CHECKING or _pygame_available:
    import pygame
else:
    pygame = None  # type: ignore[assignment]

_WINDOW_HELP = (
    "Unitree G1 keyboard teleop - keep this window focused",
    "arrows: head pan/tilt      q / e: close left / right hand",
    "w a s d: left stick         i j k l: right stick",
    "t / g: raise / lower base height",
    "y / n / r: success / failure / rerecord   space: intervene",
    "Esc: stop",
)
_PYGAME_KEY_NAMES = {"escape": "esc", "return": "enter"}

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


_ACTIVE_INPUTS: set[UnitreeG1KeyboardInput] = set()
_ACTIVE_LOCK = threading.Lock()


def external_key_event(name: str, hold_s: float = 0.35) -> None:
    """Feed a key press from another in-process source (e.g. the MuJoCo viewer window).

    Wayland sessions hide keystrokes from `pynput`, so the simulator forwards its GLFW key
    events here instead. Each call marks `name` as held for `hold_s`; key-repeat events refresh
    the timer, so a physically held key behaves like a held key and a tap like a short press.
    """
    with _ACTIVE_LOCK:
        inputs = list(_ACTIVE_INPUTS)
    for kb in inputs:
        kb.tap(name, hold_s)


class UnitreeG1KeyboardInput:
    """Keyboard input exposing the same interface as `UnitreeG1GamepadInput`.

    Keys come from `pynput` when it can capture (X11) and/or from `external_key_event`
    (in-process sources such as the simulator's viewer window).
    """

    def __init__(self, config: UnitreeG1KeyboardTeleopConfig):
        self.config = config
        self._pressed: set[str] = set()
        self._tap_expiry: dict[str, float] = {}
        self._lock = threading.Lock()
        self._listener = None
        self._window = None
        self._font = None
        self.running = True
        self.intervention_flag = False
        self.episode_end_status: TeleopEvents | None = None

    def start(self) -> None:
        backend = self.config.backend
        if backend == "external":
            with _ACTIVE_LOCK:
                _ACTIVE_INPUTS.add(self)
            return
        if backend == "window":
            self._open_window()
            return
        if backend != "pynput":
            raise ValueError(f"Unknown keyboard backend {backend!r}; use 'window', 'pynput' or 'external'")
        if pynput_keyboard is None or not pynput_can_capture():
            logging.warning(
                "pynput cannot capture keys in this session (Wayland/headless). Keys are only taken "
                "from in-process sources such as the MuJoCo viewer window (focus it and type)."
            )
            return
        self._listener = pynput_keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        with _ACTIVE_LOCK:
            _ACTIVE_INPUTS.discard(self)
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._window is not None:
            pygame.display.quit()
            self._window = None
        self.running = False

    def _open_window(self) -> None:
        """Open the dedicated key-capture window (pygame/SDL: works on X11 and Wayland)."""
        if not _pygame_available:
            logging.warning(
                "pygame is not installed; keyboard teleop has no input. pip install 'lerobot[gamepad]'"
            )
            return
        pygame.init()
        self._window = pygame.display.set_mode(self.config.window_size)
        pygame.display.set_caption("Unitree G1 teleop keys")
        self._font = pygame.font.SysFont(None, 22)
        self._draw_window()

    def _draw_window(self) -> None:
        self._window.fill((25, 25, 30))
        for i, line in enumerate(_WINDOW_HELP):
            self._window.blit(self._font.render(line, True, (230, 230, 230)), (12, 14 + 26 * i))
        with self._lock:
            held = " ".join(sorted(self._pressed))
        self._window.blit(
            self._font.render(f"held: {held}", True, (120, 220, 120)), (12, 14 + 26 * len(_WINDOW_HELP) + 14)
        )
        pygame.display.flip()

    def _pump_window(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                name = pygame.key.name(event.key)
                name = _PYGAME_KEY_NAMES.get(name, name)
                if event.type == pygame.KEYDOWN:
                    self.press(name)
                else:
                    self.release(name)
        self._draw_window()

    def update(self) -> None:
        if self._window is not None:
            self._pump_window()
        now = time.monotonic()
        with self._lock:
            expired = [name for name, until in self._tap_expiry.items() if until <= now]
        for name in expired:
            with self._lock:
                self._tap_expiry.pop(name, None)
            self.release(name)

    def tap(self, name: str, hold_s: float = 0.35) -> None:
        """Hold `name` for `hold_s` seconds (refreshed on repeat), then release it in `update()`."""
        with self._lock:
            already = name in self._tap_expiry
            self._tap_expiry[name] = time.monotonic() + hold_s
        if not already:
            self.press(name)

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
        if index == layout.trigger_left:
            return 1.0 if "t" in pressed else 0.0
        if index == layout.trigger_right:
            return 1.0 if "g" in pressed else 0.0
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
