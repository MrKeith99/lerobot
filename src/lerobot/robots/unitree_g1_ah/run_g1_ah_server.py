#!/usr/bin/env python3

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

"""DDS-to-ZMQ bridge server for the UnitreeG1Ah robot: G1 body (via `run_g1_server`) plus the
Dynamixel head and Feetech hands bridged over their own ZMQ ports.

Only runs on the Jetson attached to the robot; `unitree_sdk2py` is a hard runtime
dependency here, mirroring `unitree_g1.run_g1_server`.
"""

import argparse
import contextlib
import threading
import time

import zmq
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

from lerobot.cameras.zmq.image_server import ImageServer
from lerobot.robots.unitree_g1.run_g1_server import (
    LOWCMD_PORT,
    LOWSTATE_PORT,
    cmd_forward_loop,
    kTopicLowCommand_Debug,
    kTopicLowState,
    state_forward_loop,
)
from lerobot.robots.unitree_g1_ah.g1_ah_devices import DEFAULT_HAND_PORT, DEFAULT_HEAD_PORT, HeadHandDevice
from lerobot.robots.unitree_g1_ah.g1_ah_zmq import HEADHAND_CMD_PORT, HEADHAND_STATE_PORT, HeadHandServer


def build_camera_config(args: argparse.Namespace) -> dict:
    """Build the `ImageServer` camera config dict from CLI args."""
    if args.camera_type == "intelrealsense":
        camera_entry = {
            "type": "intelrealsense",
            "serial_number_or_name": args.camera_serial,
            "shape": [args.camera_height, args.camera_width],
        }
    else:
        camera_entry = {
            "type": "opencv",
            "device_id": args.camera_device,
            "shape": [args.camera_height, args.camera_width],
        }
    return {"fps": args.camera_fps, "cameras": {"head_camera": camera_entry}}


def main() -> None:
    parser = argparse.ArgumentParser(description="DDS-to-ZMQ bridge server for UnitreeG1Ah")
    parser.add_argument("--camera", action="store_true", help="Also launch camera server")
    parser.add_argument("--camera-type", choices=["opencv", "intelrealsense"], default="intelrealsense")
    parser.add_argument("--camera-serial", default=None, help="RealSense serial number or name")
    parser.add_argument("--camera-device", type=int, default=4, help="OpenCV camera device ID")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--head-port", default=DEFAULT_HEAD_PORT)
    parser.add_argument("--hand-port", default=DEFAULT_HAND_PORT)
    parser.add_argument("--headhand-rate", type=float, default=50.0)
    parser.add_argument("--headhand-state-port", type=int, default=HEADHAND_STATE_PORT)
    parser.add_argument("--headhand-cmd-port", type=int, default=HEADHAND_CMD_PORT)
    parser.add_argument("--no-headhand", action="store_true", help="Disable the head/hand bridge")
    parser.add_argument("--no-handshake", action="store_true", help="Skip head/hand connect handshake")
    args = parser.parse_args()
    if args.camera and args.camera_type == "intelrealsense" and not args.camera_serial:
        parser.error("--camera-serial is required with --camera-type intelrealsense")

    camera_thread = None
    if args.camera:
        camera_config = build_camera_config(args)
        camera_server = ImageServer(camera_config, port=args.camera_port)
        camera_thread = threading.Thread(target=camera_server.run, daemon=True)
        camera_thread.start()
        print(f"Camera server started on port {args.camera_port} ({args.camera_type})")

    ChannelFactoryInitialize(0)

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    status, result = msc.CheckMode()
    while result is not None and "name" in result and result["name"]:
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1.0)

    crc = CRC()

    lowcmd_pub_debug = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
    lowcmd_pub_debug.Init()

    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    first_state = None
    deadline = time.time() + 5.0
    while first_state is None and time.time() < deadline:
        first_state = lowstate_sub.Read()
        if first_state is None:
            time.sleep(0.05)
    if first_state is None:
        print("warning: no lowstate received within 5 s; mode_machine unknown")
    else:
        print(f"mode_machine={first_state.mode_machine}")

    ctx = zmq.Context.instance()

    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    state_period = 0.002  # ~500 hz
    shutdown_event = threading.Event()

    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, state_period, shutdown_event),
    )
    t_state.start()

    device = None
    headhand_server = None
    t_headhand = None
    if not args.no_headhand:
        device = HeadHandDevice(args.head_port, args.hand_port)
        device.connect(handshake=not args.no_handshake)
        device.configure()
        headhand_server = HeadHandServer(
            device,
            state_port=args.headhand_state_port,
            cmd_port=args.headhand_cmd_port,
            rate_hz=args.headhand_rate,
        )
        t_headhand = threading.Thread(target=headhand_server.run, args=(shutdown_event,), daemon=True)
        t_headhand.start()
        print(f"Head/hand bridge started (state={args.headhand_state_port}, cmd={args.headhand_cmd_port})")

    print("bridge running (lowstate -> zmq, lowcmd -> dds)")

    try:
        cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, crc)
    except KeyboardInterrupt:
        print("shutting down bridge...")
    finally:
        shutdown_event.set()
        ctx.term()
        t_state.join(timeout=2.0)
        if t_headhand is not None:
            t_headhand.join(timeout=2.0)
        if headhand_server is not None:
            headhand_server.stop()
        if device is not None:
            with contextlib.suppress(Exception):
                device.disconnect()
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
