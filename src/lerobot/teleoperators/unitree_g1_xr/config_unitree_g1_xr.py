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

from lerobot.robots.unitree_g1.end_effectors import END_EFFECTORS
from lerobot.robots.unitree_g1.g1_utils import BODIES
from lerobot.robots.unitree_g1.heads import HEAD_MOUNTS, HEAD_SENSORS

from ..config import TeleoperatorConfig

INPUT_MODES = ("controller", "hand")
DISPLAY_MODES = ("pass-through", "ego", "immersive")


@TeleoperatorConfig.register_subclass("unitree_g1_xr")
@dataclass
class UnitreeG1XRTeleopConfig(TeleoperatorConfig):
    """XR headset (e.g. Quest 3) teleoperator for the Unitree G1, built on the xr_teleoperate fork.

    `body`, `end_effector`, `head_mount` and `head_sensor` take the same values as `--robot.*` and must
    match the robot (checked on its first observation). The headset browser opens
    `https://<host>:8012` (Vuer); `cert_file`/`key_file` default to televuer's lookup
    (`$XR_TELEOP_CERT`/`$XR_TELEOP_KEY`, then `~/.config/xr_teleoperate/`).
    """

    body: str = "23dof"
    end_effector: str = "amazing_hand"
    head_mount: str = "pan_tilt"
    head_sensor: str = "d455"
    input_mode: str = "controller"
    # "pass-through" shows the room; "ego" a small head camera window in it; "immersive" the camera full view.
    display_mode: str = "pass-through"
    display_camera: str = "head_camera"
    display_image_shape: tuple[int, int] = (480, 640)
    display_fps: float = 30.0
    cert_file: str | None = None
    key_file: str | None = None
    # Local sim checkout for the arm kinematics; default: the sim's Hub snapshot (MJCF only).
    sim_root: str | None = None
    max_arm_speed_rad_s: float = 3.0
    # Arm speed after engaging, until the arms reach the operator's pose.
    engage_arm_speed_rad_s: float = 0.5
    head_speed_rad_s: float = 4.0
    hand_blend_per_s: float = 4.0
    invert_head_pitch: bool = False
    stick_deadzone: float = 0.1
    # Both grips held this long while disengaged toggle the sim elastic band.
    band_toggle_hold_s: float = 1.0
    # B held this long while disengaged resets the sim to its start pose on the elastic band.
    sim_reset_hold_s: float = 2.0
    # Sim only: disengage and reset the sim at the end of every episode (the robot ignores the reset on
    # hardware or with --robot.sim_reset_button=null).
    reset_sim_on_episode_end: bool = True

    def __post_init__(self) -> None:
        if self.band_toggle_hold_s <= 0:
            raise ValueError(f"band_toggle_hold_s must be > 0, got {self.band_toggle_hold_s}")
        if self.sim_reset_hold_s <= 0:
            raise ValueError(f"sim_reset_hold_s must be > 0, got {self.sim_reset_hold_s}")
        for name, value, allowed in (
            ("body", self.body, BODIES),
            ("end_effector", self.end_effector, END_EFFECTORS),
            ("head_mount", self.head_mount, HEAD_MOUNTS),
            ("head_sensor", self.head_sensor, HEAD_SENSORS),
            ("input_mode", self.input_mode, INPUT_MODES),
            ("display_mode", self.display_mode, DISPLAY_MODES),
        ):
            if value not in allowed:
                raise ValueError(f"Unknown {name} {value!r}; expected one of {allowed}")
