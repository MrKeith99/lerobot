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

"""Tests for `ImageServer`'s per-camera type dispatch (opencv vs intelrealsense). No hardware."""

import socket
from unittest.mock import MagicMock, patch

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.zmq import image_server as srv


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_default_camera_builds_opencv():
    with (
        patch.object(srv, "OpenCVCamera") as mock_opencv_cls,
        patch.object(srv, "RealSenseCamera") as mock_rs_cls,
    ):
        mock_opencv_cls.return_value = MagicMock()
        config = {"fps": 30, "cameras": {"cam": {"device_id": 4, "shape": [480, 640]}}}

        server = srv.ImageServer(config, port=_free_port())

        mock_opencv_cls.assert_called_once()
        mock_rs_cls.assert_not_called()
        cam_config = mock_opencv_cls.call_args[0][0]
        assert cam_config.index_or_path == 4
        assert cam_config.width == 640
        assert cam_config.height == 480

        server.socket.close()
        server.context.term()


def test_intelrealsense_camera_builds_realsense_config():
    with (
        patch.object(srv, "OpenCVCamera") as mock_opencv_cls,
        patch.object(srv, "RealSenseCamera") as mock_rs_cls,
    ):
        mock_rs_cls.return_value = MagicMock()
        config = {
            "fps": 30,
            "cameras": {
                "cam": {
                    "type": "intelrealsense",
                    "serial_number_or_name": "123456",
                    "shape": [480, 640],
                }
            },
        }

        server = srv.ImageServer(config, port=_free_port())

        mock_rs_cls.assert_called_once()
        mock_opencv_cls.assert_not_called()
        rs_config = mock_rs_cls.call_args[0][0]
        assert isinstance(rs_config, RealSenseCameraConfig)
        assert rs_config.serial_number_or_name == "123456"
        assert rs_config.width == 640
        assert rs_config.height == 480
        assert rs_config.use_rgb is True
        assert rs_config.use_depth is False

        server.socket.close()
        server.context.term()
