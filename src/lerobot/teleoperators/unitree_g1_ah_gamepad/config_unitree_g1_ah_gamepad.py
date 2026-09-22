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

_PRESETS = ("dualshock4_hidapi", "dualshock4_kernel", "xbox")


@dataclass
class GamepadLayout:
    """pygame axis/button/hat indices. Defaults = DualShock 4 under SDL2's HIDAPI driver on Linux.

    `dpad_up`/`dpad_down`/`dpad_left`/`dpad_right` are optional D-pad-as-buttons indices,
    used instead of `hat` when set (SDL's HIDAPI PS4 driver reports no hat).
    """

    left_x: int = 0
    left_y: int = 1
    right_x: int = 2
    right_y: int = 3
    button_a: int = 0
    button_b: int = 1
    button_x: int = 2
    button_y: int = 3
    button_lb: int = 9
    button_rb: int = 10
    button_back: int = 4
    button_start: int = 6
    hat: int | None = None
    dpad_up: int | None = 11
    dpad_down: int | None = 12
    dpad_left: int | None = 13
    dpad_right: int | None = 14

    @classmethod
    def dualshock4_hidapi(cls) -> GamepadLayout:
        """DualShock 4 under SDL2's HIDAPI PS4 driver (default on SDL2 >= 2.0.14). No hat; D-pad is buttons."""
        return cls(
            left_x=0,
            left_y=1,
            right_x=2,
            right_y=3,
            button_a=0,
            button_b=1,
            button_x=2,
            button_y=3,
            button_lb=9,
            button_rb=10,
            button_back=4,
            button_start=6,
            hat=None,
            dpad_up=11,
            dpad_down=12,
            dpad_left=13,
            dpad_right=14,
        )

    @classmethod
    def dualshock4_kernel(cls) -> GamepadLayout:
        """DualShock 4 under the kernel `hid-sony` joystick driver. D-pad is hat 0."""
        return cls(
            left_x=0,
            left_y=1,
            right_x=3,
            right_y=4,
            button_a=0,
            button_b=1,
            button_x=3,
            button_y=2,
            button_lb=4,
            button_rb=5,
            button_back=8,
            button_start=9,
            hat=0,
            dpad_up=None,
            dpad_down=None,
            dpad_left=None,
            dpad_right=None,
        )

    @classmethod
    def xbox(cls) -> GamepadLayout:
        """Xbox-style controller under SDL2 on Linux (the previous default). D-pad is hat 0."""
        return cls(
            left_x=0,
            left_y=1,
            right_x=3,
            right_y=4,
            button_a=0,
            button_b=1,
            button_x=2,
            button_y=3,
            button_lb=4,
            button_rb=5,
            button_back=6,
            button_start=7,
            hat=0,
            dpad_up=None,
            dpad_down=None,
            dpad_left=None,
            dpad_right=None,
        )


def _layout_for_preset(preset: str) -> GamepadLayout:
    if preset == "dualshock4_hidapi":
        return GamepadLayout.dualshock4_hidapi()
    if preset == "dualshock4_kernel":
        return GamepadLayout.dualshock4_kernel()
    if preset == "xbox":
        return GamepadLayout.xbox()
    raise ValueError(f"Unknown gamepad preset {preset!r}; expected one of {_PRESETS}")


@TeleoperatorConfig.register_subclass("unitree_g1_ah_gamepad")
@dataclass
class UnitreeG1AhGamepadTeleopConfig(TeleoperatorConfig):
    """Hardware-light gamepad teleoperator for the UnitreeG1Ah robot.

    Emits every key in `TELEOP_ACTION_KEYS`: held body/hand poses, D-pad-driven
    head targets, an R1/L1 (RB/LB on Xbox)-blended hand open/close, and the 4
    `REMOTE_AXES` driven by the sticks (zeros when `emit_remote_axes` is False).
    """

    layout: GamepadLayout = field(default_factory=GamepadLayout)
    preset: str = "dualshock4_hidapi"
    deadzone: float = 0.1
    head_speed_rad_s: float = 0.8
    hand_blend_per_s: float = 2.0
    emit_remote_axes: bool = True
    invert_tilt: bool = False
    initial_positions: dict[str, float] | None = None

    def __post_init__(self) -> None:
        preset_layout = _layout_for_preset(self.preset)
        if self.layout == GamepadLayout():
            self.layout = preset_layout
