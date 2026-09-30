#!/usr/bin/env python3

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
DDS-to-ZMQ bridge server for Unitree G1 robot.

This server runs on the robot and forwards:
- Robot state (LowState) from DDS to ZMQ (for remote clients)
- Robot commands (LowCmd) from ZMQ to DDS (from remote clients)
- Optionally, the Dex3 hand / Dex1 gripper command and state topics (`--hands`)
- Optionally, the pan/tilt head and AmazingHand servos over their own ZMQ bridge (`--headhand`)
- Optionally, a head camera stream (`--camera`)

Uses JSON for secure serialization instead of pickle.
"""

import argparse
import base64
import contextlib
import json
import threading
import time
from typing import Any

import zmq
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__MotorCmd_,
    unitree_hg_msg_dds__HandCmd_,
    unitree_hg_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    HandCmd_,
    HandState_,
    LowCmd_ as hg_LowCmd,
    LowState_ as hg_LowState,
)
from unitree_sdk2py.utils.crc import CRC

from lerobot.cameras.zmq.image_server import ImageServer
from lerobot.robots.unitree_g1.dex_hands import dex_topics
from lerobot.robots.unitree_g1.headhand_devices import DEFAULT_HAND_PORT, DEFAULT_HEAD_PORT, HeadHandDevice
from lerobot.robots.unitree_g1.headhand_zmq import HEADHAND_CMD_PORT, HEADHAND_STATE_PORT, HeadHandServer
from lerobot.robots.unitree_g1.unitree_sdk2_socket import (
    LOWCMD_PORT,
    LOWSTATE_PORT,
    kTopicLowCommand_Debug,
    kTopicLowState,
    topic_port,
)

NUM_MOTORS = 35


def lowstate_to_dict(msg: hg_LowState) -> dict[str, Any]:
    """Convert LowState SDK message to a JSON-serializable dictionary."""
    motor_states = []
    for i in range(NUM_MOTORS):
        temp = msg.motor_state[i].temperature
        avg_temp = float(sum(temp) / len(temp)) if isinstance(temp, list) else float(temp)
        motor_states.append(
            {
                "q": float(msg.motor_state[i].q),
                "dq": float(msg.motor_state[i].dq),
                "tau_est": float(msg.motor_state[i].tau_est),
                "temperature": avg_temp,
            }
        )

    return {
        "motor_state": motor_states,
        "imu_state": {
            "quaternion": [float(x) for x in msg.imu_state.quaternion],
            "gyroscope": [float(x) for x in msg.imu_state.gyroscope],
            "accelerometer": [float(x) for x in msg.imu_state.accelerometer],
            "rpy": [float(x) for x in msg.imu_state.rpy],
            "temperature": float(msg.imu_state.temperature),
        },
        # Encode bytes as base64 for JSON compatibility
        "wireless_remote": base64.b64encode(bytes(msg.wireless_remote)).decode("ascii"),
        "mode_machine": int(msg.mode_machine),
    }


def dict_to_lowcmd(data: dict[str, Any]) -> hg_LowCmd:
    """Convert dictionary back to LowCmd SDK message."""
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = data.get("mode_pr", 0)
    cmd.mode_machine = data.get("mode_machine", 0)

    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)

    return cmd


def state_forward_loop(
    lowstate_sub: ChannelSubscriber,
    lowstate_sock: zmq.Socket,
    state_period: float,
    shutdown_event: threading.Event,
) -> None:
    """Read observation from DDS and forward to ZMQ clients."""
    last_state_time = 0.0

    while not shutdown_event.is_set():
        # read from DDS
        msg = lowstate_sub.Read()
        if msg is None:
            continue

        now = time.time()
        # optional downsampling (if robot dds rate > state_period)
        if now - last_state_time >= state_period:
            # Convert to dict and serialize with JSON
            state_dict = lowstate_to_dict(msg)
            payload = json.dumps({"topic": kTopicLowState, "data": state_dict}).encode("utf-8")
            # if no subscribers / tx buffer full, just drop
            with contextlib.suppress(zmq.Again):
                lowstate_sock.send(payload, zmq.NOBLOCK)
            last_state_time = now


def cmd_forward_loop(
    lowcmd_sock: zmq.Socket,
    lowcmd_pub_debug: ChannelPublisher,
    crc: CRC,
) -> None:
    """Receive commands from ZMQ and forward to DDS."""
    while True:
        try:
            payload = lowcmd_sock.recv()
        except zmq.ContextTerminated:
            break
        msg_dict = json.loads(payload.decode("utf-8"))

        topic = msg_dict.get("topic", "")
        cmd_data = msg_dict.get("data", {})

        # Reconstruct LowCmd object from dict
        cmd = dict_to_lowcmd(cmd_data)

        # recompute crc
        cmd.crc = crc.Crc(cmd)

        if topic == kTopicLowCommand_Debug:
            lowcmd_pub_debug.Write(cmd)


def hand_state_to_dict(msg: HandState_ | MotorStates_) -> dict[str, Any]:
    """Convert a Dex3 `HandState_` or Dex1 `MotorStates_` to a JSON-serializable dictionary."""
    states = msg.motor_state if hasattr(msg, "motor_state") else msg.states
    return {
        "motor_state": [{"q": float(m.q), "dq": float(m.dq), "tau_est": float(m.tau_est)} for m in states]
    }


def dict_to_hand_cmd(family: str, data: dict[str, Any]) -> HandCmd_ | MotorCmds_:
    """Convert a dictionary back to a Dex3 `HandCmd_` (family "dex3") or Dex1 `MotorCmds_` ("dex1")."""
    motor_data = data.get("motor_cmd", [])
    if family == "dex3":
        cmd = unitree_hg_msg_dds__HandCmd_()
        motors = cmd.motor_cmd
    else:
        cmd = MotorCmds_(cmds=[unitree_go_msg_dds__MotorCmd_() for _ in motor_data])
        motors = cmd.cmds
    for motor, values in zip(motors, motor_data, strict=False):
        for field in ("mode", "q", "dq", "kp", "kd", "tau"):
            setattr(motor, field, values.get(field, 0))
    return cmd


def start_hand_bridge(family: str, ctx: zmq.Context) -> list[threading.Thread]:
    """Bridge both sides' Dex3 (`family="dex3"`) or Dex1 (`"dex1"`) topics: state DDS -> ZMQ, command ZMQ -> DDS."""
    cmd_type, state_type = (HandCmd_, HandState_) if family == "dex3" else (MotorCmds_, MotorStates_)
    threads = []
    for side in ("left", "right"):
        cmd_topic, state_topic = dex_topics(family, False, side)

        state_sock = ctx.socket(zmq.PUB)
        state_sock.bind(f"tcp://0.0.0.0:{topic_port(state_topic)}")

        def forward_state(msg, sock=state_sock, topic=state_topic) -> None:
            payload = json.dumps({"topic": topic, "data": hand_state_to_dict(msg)}).encode("utf-8")
            with contextlib.suppress(zmq.Again, zmq.ContextTerminated):
                sock.send(payload, zmq.NOBLOCK)

        state_sub = ChannelSubscriber(state_topic, state_type)
        state_sub.Init(forward_state, 1)

        cmd_pub = ChannelPublisher(cmd_topic, cmd_type)
        cmd_pub.Init()
        cmd_sock = ctx.socket(zmq.PULL)
        cmd_sock.bind(f"tcp://0.0.0.0:{topic_port(cmd_topic)}")

        def forward_cmds(sock=cmd_sock, pub=cmd_pub) -> None:
            while True:
                try:
                    payload = sock.recv()
                except zmq.ContextTerminated:
                    break
                pub.Write(dict_to_hand_cmd(family, json.loads(payload.decode("utf-8")).get("data", {})))

        thread = threading.Thread(target=forward_cmds, daemon=True)
        thread.start()
        threads.append(thread)
    return threads


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
    """Main entry point for the robot server bridge."""
    parser = argparse.ArgumentParser(description="DDS-to-ZMQ bridge server for Unitree G1")
    parser.add_argument("--camera", action="store_true", help="Also launch camera server")
    parser.add_argument("--camera-type", choices=["opencv", "intelrealsense"], default="opencv")
    parser.add_argument("--camera-serial", default=None, help="RealSense serial number or name")
    parser.add_argument("--camera-device", type=int, default=4, help="OpenCV camera device ID (default: 4)")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera FPS (default: 30)")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--camera-port", type=int, default=5555, help="Camera ZMQ port (default: 5555)")
    parser.add_argument(
        "--hands", choices=["none", "dex3", "dex1"], default="none", help="Also bridge Dex3/Dex1 topics"
    )
    parser.add_argument(
        "--headhand", action="store_true", help="Also run the pan/tilt head + AmazingHand bridge"
    )
    parser.add_argument("--no-head", action="store_true", help="Head/hand bridge without the head bus")
    parser.add_argument("--no-hands", action="store_true", help="Head/hand bridge without the hand bus")
    parser.add_argument("--head-port", default=DEFAULT_HEAD_PORT)
    parser.add_argument("--hand-port", default=DEFAULT_HAND_PORT)
    parser.add_argument("--headhand-rate", type=float, default=50.0)
    parser.add_argument("--headhand-state-port", type=int, default=HEADHAND_STATE_PORT)
    parser.add_argument("--headhand-cmd-port", type=int, default=HEADHAND_CMD_PORT)
    parser.add_argument("--no-handshake", action="store_true", help="Skip head/hand connect handshake")
    args = parser.parse_args()
    if args.camera and args.camera_type == "intelrealsense" and not args.camera_serial:
        parser.error("--camera-serial is required with --camera-type intelrealsense")
    if args.headhand and args.no_head and args.no_hands:
        parser.error("--headhand needs the head bus, the hand bus or both")

    # Optionally start camera server in background thread
    camera_thread = None
    if args.camera:
        camera_server = ImageServer(build_camera_config(args), port=args.camera_port)
        camera_thread = threading.Thread(target=camera_server.run, daemon=True)
        camera_thread.start()
        print(f"Camera server started on port {args.camera_port} ({args.camera_type})")

    # initialize DDS
    ChannelFactoryInitialize(0)

    # stop all active publishers on the robot
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    status, result = msc.CheckMode()
    while result is not None and "name" in result and result["name"]:
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1.0)

    crc = CRC()

    # initialize DDS publisher
    lowcmd_pub_debug = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
    lowcmd_pub_debug.Init()

    # initialize DDS subscriber
    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    # report mode_machine, which the 23dof client checks against its --robot.revision
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

    # initialize ZMQ
    ctx = zmq.Context.instance()

    # receive commands from remote client
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    # publish state to remote clients
    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    state_period = 0.002  # ~500 hz
    shutdown_event = threading.Event()

    # start observation forwarding in background thread
    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, state_period, shutdown_event),
    )
    t_state.start()

    if args.hands != "none":
        start_hand_bridge(args.hands, ctx)
        print(f"{args.hands} hand bridge started")

    device = None
    headhand_server = None
    t_headhand = None
    if args.headhand:
        device = HeadHandDevice(
            None if args.no_head else args.head_port, None if args.no_hands else args.hand_port
        )
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

    # run command forwarding in main thread
    try:
        cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, crc)
    except KeyboardInterrupt:
        print("shutting down bridge...")
    finally:
        shutdown_event.set()
        ctx.term()  # terminates blocking zmq.recv() calls
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
