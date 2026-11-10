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
import logging
import multiprocessing
import signal
import threading
import time
from typing import Any

import zmq
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
from lerobot.robots.unitree_g1.hand_force import add_hand_force_args, hand_force_from_args
from lerobot.robots.unitree_g1.handover import (
    DEFAULT_HANDOVER_FSM,
    HANDOVER_STRATEGIES,
    Handover,
    HandoverState,
    make_strategy,
    relay_state,
)
from lerobot.robots.unitree_g1.headhand_devices import (
    DEFAULT_HAND_PORT,
    DEFAULT_HEAD_PORT,
    HEAD_GOAL_CURRENT,
    HeadHandDevice,
)
from lerobot.robots.unitree_g1.headhand_zmq import HEADHAND_CMD_PORT, HEADHAND_STATE_PORT, HeadHandServer
from lerobot.robots.unitree_g1.safety import (
    ESTOP_KEYS,
    LOWCMD_TIMEOUT_S,
    LowCmdWatchdog,
    damping_motor_cmds,
    remote_key_mask,
    remote_keys,
)
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
    handover: Handover | None = None,
    watchdog: LowCmdWatchdog | None = None,
    estop_mask: int = 0,
) -> None:
    """Read observation from DDS and forward to ZMQ clients (with the relay's handover state). Every lowstate
    with all `estop_mask` remote keys held latches damping for the active client (remote e-stop)."""
    last_state_time = 0.0

    while not shutdown_event.is_set():
        # read from DDS
        msg = lowstate_sub.Read()
        if msg is None:
            continue

        if (
            estop_mask
            and watchdog is not None
            and remote_keys(msg.wireless_remote) & estop_mask == estop_mask
        ):
            session = watchdog.estop()
            if session is not None:
                print(f"remote e-stop: DAMPING session {session} (restart the client to resume)", flush=True)

        now = time.time()
        # optional downsampling (if robot dds rate > state_period)
        if now - last_state_time >= state_period:
            # Convert to dict and serialize with JSON
            state_dict = lowstate_to_dict(msg)
            if handover is not None:
                state_dict["relay_handover"] = handover.state.value
                state_dict["relay_state"] = relay_state(
                    watchdog is not None and watchdog.forwarding, handover
                )
            payload = json.dumps({"topic": kTopicLowState, "data": state_dict}).encode("utf-8")
            # if no subscribers / tx buffer full, just drop
            with contextlib.suppress(zmq.Again):
                lowstate_sock.send(payload, zmq.NOBLOCK)
            last_state_time = now


def damping_lowcmd(mode_machine: int) -> hg_LowCmd:
    """LowCmd putting every body motor in damping (see `safety.py`)."""
    return dict_to_lowcmd({"mode_machine": mode_machine, "motor_cmd": damping_motor_cmds()})


def publish_lowcmd(pub: ChannelPublisher, crc: CRC, cmd: hg_LowCmd) -> None:
    cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)


def cmd_forward_loop(
    lowcmd_sock: zmq.Socket,
    lowcmd_pub_debug: ChannelPublisher,
    crc: CRC,
    watchdog: LowCmdWatchdog,
    mode_machine: int,
    damping_period: float = 0.02,
    shutdown_event: threading.Event | None = None,
    handover: Handover | None = None,
) -> None:
    """Forward client lowcmds from ZMQ to DDS while the watchdog allows it; otherwise hold damping.

    A lowcmd flagged `handover` asks to give the robot back to Unitree's controller: the relay keeps publishing
    the client's latest (stand) command, watchdog suspended, while `handover` switches; once done it publishes
    nothing more. A normal lowcmd from the client cancels a handover in progress or failed.
    """
    handover = handover if handover is not None else Handover(None)
    poller = zmq.Poller()
    poller.register(lowcmd_sock, zmq.POLLIN)
    last_damping_t = last_hold_t = float("-inf")
    hold_cmd: hg_LowCmd | None = None
    rejected_session: str | None = None
    print("lowcmd watchdog: damping until a client connects")
    while shutdown_event is None or not shutdown_event.is_set():
        try:
            ready = dict(poller.poll(timeout=int(damping_period * 1000)))
        except zmq.ContextTerminated:
            break
        now = time.monotonic()
        if lowcmd_sock in ready:
            try:
                payload = lowcmd_sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                payload = None
            except zmq.ContextTerminated:
                break
            if payload is not None:
                msg_dict = json.loads(payload.decode("utf-8"))
                session = msg_dict.get("session")
                wants_handover = bool(msg_dict.get("handover"))
                cmd = dict_to_lowcmd(msg_dict.get("data", {}))
                if (
                    msg_dict.get("topic", "") != kTopicLowCommand_Debug
                    or handover.state == HandoverState.DONE
                ):
                    pass
                elif handover.active:
                    if session == watchdog.session:
                        watchdog.on_command(session, now)
                        if wants_handover:
                            hold_cmd = cmd
                        else:
                            handover.cancel()
                            print("handover: cancelled by the client, forwarding its commands again")
                            publish_lowcmd(lowcmd_pub_debug, crc, cmd)
                else:
                    was_forwarding = watchdog.forwarding
                    if watchdog.on_command(session, now):
                        if not was_forwarding:
                            print(f"lowcmd watchdog: forwarding client session {session}")
                        if wants_handover:
                            handover.request()
                            if handover.active:
                                hold_cmd = cmd
                                print("handover: requested, holding the client's command while switching")
                        publish_lowcmd(lowcmd_pub_debug, crc, cmd)
                    elif session != rejected_session:
                        rejected_session = session
                        print(
                            f"lowcmd watchdog: ignoring session {session} (timed out or not the active one)"
                        )
        if handover.state == HandoverState.DONE:
            continue
        if handover.active:
            if (
                hold_cmd is not None
                and not handover.publishing_stopped.is_set()
                and now - last_hold_t >= damping_period
            ):
                publish_lowcmd(lowcmd_pub_debug, crc, hold_cmd)
                last_hold_t = now
            continue
        was_forwarding = watchdog.forwarding
        if watchdog.check(now):
            if was_forwarding:
                print(
                    f"lowcmd watchdog: no lowcmd for {watchdog.stale_s(now):.2f}s from session "
                    f"{watchdog.session}: DAMPING (restart the client to resume)"
                )
            if now - last_damping_t >= damping_period:
                publish_lowcmd(lowcmd_pub_debug, crc, damping_lowcmd(mode_machine))
                last_damping_t = now


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


def headhand_process(device_kwargs, handshake, server_kwargs, shutdown, release, ready) -> None:
    """Head/hand bridge in its own process: in the relay's threads it waited on the GIL (lowstate JSON at 500 Hz,
    camera encoding) and ran at ~25 Hz. `shutdown`, `release` and `ready` are multiprocessing Events."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the relay stops this process through `shutdown`
    logging.basicConfig(level=logging.WARNING)
    device = HeadHandDevice(**device_kwargs)
    server = None
    try:
        device.connect(handshake=handshake)
        device.configure()
        server = HeadHandServer(device, release_event=release, **server_kwargs)
        server.bind()
        ready.set()
        server.run(shutdown)
    finally:
        if server is not None:
            server.stop()
        with contextlib.suppress(Exception):
            device.disconnect()


def start_headhand_process(device_kwargs, handshake, server_kwargs, timeout_s: float = 20.0):
    """Start `headhand_process`; returns (process, shutdown, release) once it serves, or raises."""
    mp = multiprocessing.get_context("spawn")  # never fork a process that runs DDS threads
    shutdown, release, ready = mp.Event(), mp.Event(), mp.Event()
    proc = mp.Process(
        target=headhand_process,
        args=(device_kwargs, handshake, server_kwargs, shutdown, release, ready),
        name="headhand",
        daemon=True,
    )
    proc.start()
    deadline = time.monotonic() + timeout_s
    while not ready.wait(0.1):
        if not proc.is_alive() or time.monotonic() > deadline:
            shutdown.set()
            proc.join(timeout=2.0)
            raise RuntimeError(f"head/hand process did not start (exit code {proc.exitcode})")
    return proc, shutdown, release


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
    logging.basicConfig(level=logging.INFO)
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
    parser.add_argument(
        "--head-goal-current",
        type=int,
        default=HEAD_GOAL_CURRENT,
        help="Head Goal_Current in Dynamixel units (XC330: 1 mA), the cap on how hard the head pushes (mode 5)",
    )
    parser.add_argument("--hand-port", default=DEFAULT_HAND_PORT)
    parser.add_argument("--headhand-rate", type=float, default=50.0)
    parser.add_argument("--headhand-state-port", type=int, default=HEADHAND_STATE_PORT)
    parser.add_argument("--headhand-cmd-port", type=int, default=HEADHAND_CMD_PORT)
    parser.add_argument("--no-handshake", action="store_true", help="Skip head/hand connect handshake")
    parser.add_argument(
        "--handover",
        choices=HANDOVER_STRATEGIES,
        default="release",
        help="How to take control from Unitree's locomotion service and whether a client may hand it back "
        "(experimental, test on the gantry): release (no handover), user-ctrl, select-mode",
    )
    parser.add_argument(
        "--handover-fsm",
        type=int,
        default=DEFAULT_HANDOVER_FSM,
        help="select-mode only: loco FSM id requested after the handover (default: %(default)s, walk/run)",
    )
    parser.add_argument(
        "--assume-damping",
        action="store_true",
        help="Start even though Unitree's loco service cannot confirm damping (it does not answer, e.g. after "
        "a previous relay released it). Only after checking the robot is in damping.",
    )
    parser.add_argument(
        "--lowcmd-timeout",
        type=float,
        default=LOWCMD_TIMEOUT_S,
        help="Damping once no lowcmd arrived for this many seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--estop-keys",
        default=ESTOP_KEYS,
        help="Unitree remote keys that latch damping while held, e.g. L2+B; '' turns it off (default: %(default)s)",
    )
    add_hand_force_args(parser)
    args = parser.parse_args()
    estop_mask = remote_key_mask(args.estop_keys)
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
        mode_machine = 0
    else:
        mode_machine = int(first_state.mode_machine)
        print(f"mode_machine={mode_machine}")

    # take low-level control from Unitree's locomotion service, then damp at once
    take_control, give_back = make_strategy(args.handover, args.handover_fsm, args.assume_damping)
    try:
        take_control()
    except RuntimeError as e:
        print(f"error: {e}")
        raise SystemExit(1) from e
    handover = Handover(give_back)
    publish_lowcmd(lowcmd_pub_debug, crc, damping_lowcmd(mode_machine))
    print(f"low-level control taken ({args.handover}); handover {'enabled' if handover.supported else 'off'}")

    # initialize ZMQ
    ctx = zmq.Context.instance()

    # receive commands from remote client
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    # publish state to remote clients
    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    watchdog = LowCmdWatchdog(args.lowcmd_timeout)
    state_period = 0.002  # ~500 hz
    shutdown_event = threading.Event()

    # start observation forwarding in background thread
    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, state_period, shutdown_event, handover, watchdog, estop_mask),
    )
    print(f"remote e-stop: {args.estop_keys or 'off'}")
    t_state.start()

    if args.hands != "none":
        start_hand_bridge(args.hands, ctx)
        print(f"{args.hands} hand bridge started")

    headhand = None
    if args.headhand:
        headhand = start_headhand_process(
            {
                "head_port": None if args.no_head else args.head_port,
                "hand_port": None if args.no_hands else args.hand_port,
                "hand_force": hand_force_from_args(args),
                "head_goal_current": args.head_goal_current,
            },
            not args.no_handshake,
            {
                "state_port": args.headhand_state_port,
                "cmd_port": args.headhand_cmd_port,
                "rate_hz": args.headhand_rate,
            },
        )
        watchdog.on_trip = headhand[2].set  # watchdog timeout or remote e-stop: head/hands off too
        print(
            f"Head/hand bridge started in its own process "
            f"(state={args.headhand_state_port}, cmd={args.headhand_cmd_port})"
        )

    print("bridge running (lowstate -> zmq, lowcmd -> dds)")

    # run command forwarding in main thread
    try:
        cmd_forward_loop(
            lowcmd_sock,
            lowcmd_pub_debug,
            crc,
            watchdog,
            mode_machine,
            shutdown_event=shutdown_event,
            handover=handover,
        )
    except KeyboardInterrupt:
        print("shutting down bridge...")
    finally:
        # Leave the robot in damping, never in the last client command or zero torque, unless it was handed
        # over to Unitree's controller.
        if handover.state == HandoverState.DONE:
            print("robot left with Unitree's controller (handed over)")
        else:
            for _ in range(5):
                with contextlib.suppress(Exception):
                    publish_lowcmd(lowcmd_pub_debug, crc, damping_lowcmd(mode_machine))
                time.sleep(0.01)
            print("robot left in damping")
        shutdown_event.set()
        ctx.term()  # terminates blocking zmq.recv() calls
        t_state.join(timeout=2.0)
        if headhand is not None:
            proc, headhand_shutdown, _ = headhand
            headhand_shutdown.set()
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
