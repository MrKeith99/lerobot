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

"""Status banner drawn on the image shown in the headset."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# RGB, like the camera images.
WHITE = (255, 255, 255)
GREEN = (70, 230, 100)
RED = (255, 60, 60)
YELLOW = (255, 220, 0)
ORANGE = (255, 150, 0)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
# The ego view is ~0.35x life size in the headset, so the text is big: the first line takes up to 8% of the
# image height, the others 7%, shrunk to fit the width.
_MAIN_HEIGHT_FRAC = 0.08
_SUB_HEIGHT_FRAC = 0.07
_BANNER_ALPHA = 0.65  # how much the image under the banner is darkened


@dataclass(frozen=True)
class Line:
    text: str
    color: tuple[int, int, int] = WHITE
    dot: bool = False  # a filled circle before the text (recording)


def _fit_scale(text: str, thickness_per_scale: float, max_height_px: float, max_width_px: float) -> float:
    (w, h), _ = cv2.getTextSize(text, _FONT, 1.0, max(1, round(thickness_per_scale)))
    return max(min(max_height_px / h, max_width_px / w), 0.2)


def draw_status(image: np.ndarray, lines: list[Line]) -> np.ndarray:
    """Return a copy of `image` (HxWx3 uint8) with a banner of `lines` along the top; `image` is untouched."""
    out = np.array(image, copy=True)
    if not lines or out.dtype != np.uint8 or out.ndim != 3 or out.shape[2] != 3:
        return out
    height, width = out.shape[:2]
    margin = int(0.02 * width)
    rows = []
    y = margin
    for i, line in enumerate(lines):
        frac = _MAIN_HEIGHT_FRAC if i == 0 else _SUB_HEIGHT_FRAC
        indent = int(frac * height) if line.dot else 0
        scale = _fit_scale(line.text, 2.0, frac * height, width - 2 * margin - indent)
        thickness = max(1, round(scale * 2))
        (_, text_h), baseline = cv2.getTextSize(line.text, _FONT, scale, thickness)
        rows.append((line, scale, thickness, y + text_h, text_h, indent))
        y += text_h + baseline + margin // 2
    bottom = min(y + margin // 2, height)
    out[:bottom] = (out[:bottom] * (1 - _BANNER_ALPHA)).astype(np.uint8)
    for line, scale, thickness, baseline_y, text_h, indent in rows:
        if line.dot:
            cv2.circle(out, (margin + indent // 2, baseline_y - text_h // 2), indent // 3, line.color, -1)
        cv2.putText(
            out, line.text, (margin + indent, baseline_y), _FONT, scale, line.color, thickness, cv2.LINE_AA
        )
    return out
