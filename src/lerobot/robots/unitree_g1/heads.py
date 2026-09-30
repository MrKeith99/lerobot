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

"""Head variants for the Unitree G1: the stock fixed head, or a 2-DoF Dynamixel pan/tilt D455 mount.

Pure stdlib: no hardware/SDK/bus imports here.
"""

from __future__ import annotations

HEADS: tuple[str, ...] = ("none", "d455_pan_tilt")

# d455_pan_tilt: two Dynamixel XL330-M288 servos, driven through the head/hand ZMQ bridge.
HEAD_MOTORS: dict[str, tuple[int, str]] = {
    "kHeadYaw": (1, "xl330-m288"),
    "kHeadPitch": (2, "xl330-m288"),
}
HEAD_KEYS: tuple[str, ...] = tuple(f"{name}.q" for name in HEAD_MOTORS)
HEAD_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "kHeadYaw": (-0.7, 0.7),
    "kHeadPitch": (-1.57, 0.8),
}
DEFAULT_HEAD_Q: tuple[float, float] = (0.0, 0.0)
