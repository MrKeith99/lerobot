#!/usr/bin/env python

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

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.import_utils import _unitree_sdk_available, require_package

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config
from .dex_hands import DexHandDriver
from .end_effectors import HAND_CLOSURE_KEYS, HAND_SIDES, HAND_SPECS, hand_closure_key
from .g1_kinematics import G1_29_ArmIK
from .g1_utils import (
    BASE_HEIGHT_KEY,
    MODE_MACHINE_BY_REVISION,
    NAV_KEYS,
    REMOTE_KEYS,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    default_remote_input,
    invalid_body_keys,
    invalid_sdk_slots,
    make_locomotion_controller,
)
from .headhand import LEGACY_MOTOR_NAMES, HeadHandBridge
from .heads import HEAD_KEYS

if TYPE_CHECKING or _unitree_sdk_available:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize as _SDKChannelFactoryInitialize,
        ChannelPublisher as _SDKChannelPublisher,
        ChannelSubscriber as _SDKChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowCmd_ as hg_LowCmd,
        LowState_ as hg_LowState,
    )
    from unitree_sdk2py.utils.crc import CRC
else:
    _SDKChannelFactoryInitialize = None
    _SDKChannelPublisher = None
    _SDKChannelSubscriber = None
    unitree_hg_msg_dds__LowCmd_ = None
    hg_LowCmd = None
    hg_LowState = None
    CRC = None

logger = logging.getLogger(__name__)


@runtime_checkable
class LocomotionController(Protocol):
    control_dt: float

    def run_step(self, action: dict, lowstate) -> dict: ...

    def reset(self) -> None: ...


# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"


@dataclass
class MotorState:
    q: float | None = None  # position
    dq: float | None = None  # velocity
    tau_est: float | None = None  # estimated torque
    temperature: float | None = None  # motor temperature


@dataclass
class IMUState:
    quaternion: np.ndarray | None = None  # [w, x, y, z]
    gyroscope: np.ndarray | None = None  # [x, y, z] angular velocity (rad/s)
    accelerometer: np.ndarray | None = None  # [x, y, z] linear acceleration (m/s²)
    rpy: np.ndarray | None = None  # [roll, pitch, yaw] (rad)
    temperature: float | None = None  # IMU temperature


# g1 observation class
@dataclass
class G1_29_LowState:  # noqa: N801
    motor_state: list[MotorState] = field(default_factory=lambda: [MotorState() for _ in G1_29_JointIndex])
    imu_state: IMUState = field(default_factory=IMUState)
    wireless_remote: bytes | None = None  # Raw wireless remote data
    mode_machine: int = 0  # Robot mode


@contextlib.contextmanager
def _env_vars(values: Mapping[str, str]) -> Iterator[None]:
    """Temporarily set environment variables."""
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class UnitreeG1(Robot):
    """Unitree G1 with a configurable embodiment: 29dof/23dof body, optional Dex1/Dex3/AmazingHand end
    effector and optional pan/tilt head (see `UnitreeG1Config`).

    The body is commanded over DDS lowcmd (or its ZMQ stand-in on hardware), Dex1/Dex3 over their own
    DDS topics, and the pan/tilt head and AmazingHand servos over the head/hand ZMQ bridge.
    """

    config_class = UnitreeG1Config
    name = "unitree_g1"

    def __init__(self, config: UnitreeG1Config):
        require_package("unitree-sdk2py", extra="unitree_g1", import_name="unitree_sdk2py")
        # Per-embodiment name: the dataset robot_type and the calibration file id.
        self.name = config.robot_type
        super().__init__(config)

        logger.info("Initialize UnitreeG1...")

        self.config = config
        self.control_dt = config.control_dt

        # Initialize cameras config (ZMQ-based) - actual connection in connect()
        self._cameras = make_cameras_from_configs(config.cameras)

        # Import channel classes based on mode
        if config.is_simulation:
            self._ChannelFactoryInitialize = _SDKChannelFactoryInitialize
            self._ChannelPublisher = _SDKChannelPublisher
            self._ChannelSubscriber = _SDKChannelSubscriber
        else:
            from .unitree_sdk2_socket import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )

            self._ChannelFactoryInitialize = ChannelFactoryInitialize
            self._ChannelPublisher = ChannelPublisher
            self._ChannelSubscriber = ChannelSubscriber

        # Embodiment: body slots without a motor, end effector and head/hand bridge
        self._invalid_slots = frozenset(invalid_sdk_slots(config.body))
        self._invalid_body_keys = invalid_body_keys(config.body)
        self.hand_spec = HAND_SPECS.get(config.end_effector)
        self.dex_hand = (
            DexHandDriver(
                self.hand_spec, config.is_simulation, self._ChannelPublisher, self._ChannelSubscriber
            )
            if config.end_effector in ("dex1", "dex3")
            else None
        )
        self.headhand = None
        if config.headhand_motors:
            headhand_ip = config.headhand_ip or ("127.0.0.1" if config.is_simulation else config.robot_ip)
            self.headhand = HeadHandBridge(
                config.headhand_motors,
                headhand_ip,
                config.headhand_state_port,
                config.headhand_cmd_port,
                timeout_s=config.headhand_timeout_s,
                stale_warn_s=config.headhand_stale_warn_s,
            )

        # Initialize state variables
        self.sim_env = None
        self._env_wrapper = None
        self._lowstate = None
        self._lowstate_lock = threading.Lock()
        # send_action (main thread) and the controller thread both publish lowcmd;
        # CycloneDDS delivers rt/lowcmd inline to in-process readers, so writes must not overlap.
        self._lowcmd_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.subscribe_thread = None

        self.arm_ik = G1_29_ArmIK() if config.gravity_compensation else None

        # Lower-body controller loaded dynamically
        self.controller: LocomotionController | None = make_locomotion_controller(config.controller)

        # Controller thread state
        self._controller_thread = None
        self._controller_action_lock = threading.Lock()
        self.controller_input = default_remote_input()
        self.controller_output = {}
        self._band_button_was_pressed = False
        self._reset_button_was_pressed = False
        self._band_was_attached = False

    def _sim_band_attached(self) -> bool:
        """True while the simulated robot hangs on the MuJoCo elastic band."""
        sim = getattr(getattr(self.sim_env, "simulator", None), "sim_env", None)
        band = getattr(sim, "elastic_band", None)
        return band is not None and band.enable

    def _make_sim_legs_limp(self) -> None:
        """Zero legs/waist gains in the command send_action keeps republishing and in the one the sim holds."""
        msg = getattr(self, "msg", None)
        if msg is not None:
            with self._lowcmd_lock:
                for motor in range(15):  # legs + waist (controller-owned joints)
                    msg.motor_cmd[motor].kp = 0.0
                    msg.motor_cmd[motor].kd = 0.0
                    msg.motor_cmd[motor].tau = 0.0
        sim = getattr(getattr(self.sim_env, "simulator", None), "sim_env", None)
        bridge = getattr(sim, "unitree_bridge", None)
        if bridge is not None:
            with bridge.low_cmd_lock:
                for motor_cmd in bridge.low_cmd.motor_cmd[:15]:
                    motor_cmd.kp = 0.0
                    motor_cmd.kd = 0.0
                    motor_cmd.tau = 0.0

    def _poll_sim_gamepad_buttons(self):
        """Sim-only gamepad shortcuts: toggle the elastic band (same as "9" in the MuJoCo viewer) and reset."""
        sim = getattr(getattr(self.sim_env, "simulator", None), "sim_env", None)
        joystick = getattr(getattr(sim, "unitree_bridge", None), "joystick", None)
        band = getattr(sim, "elastic_band", None)
        if joystick is None or band is None:
            return

        def just_pressed(button: int | None, was_pressed_attr: str) -> bool:
            if button is None or button >= joystick.get_numbuttons():
                return False
            pressed = bool(joystick.get_button(button))
            was_pressed = getattr(self, was_pressed_attr)
            setattr(self, was_pressed_attr, pressed)
            return pressed and not was_pressed

        if just_pressed(self.config.sim_band_toggle_button, "_band_button_was_pressed"):
            band.enable = not band.enable
            logger.info(f"Elastic band {'attached' if band.enable else 'released'}")

        if just_pressed(self.config.sim_reset_button, "_reset_button_was_pressed"):
            # Runs on the sim-stepping thread, so MuJoCo state is never modified mid-step
            band.enable = True
            band.length = 0
            self.sim_env.reset()
            # Legs go limp right away; the controller loop keeps the policy off until the band is released
            self._make_sim_legs_limp()
            logger.info("Simulation reset: robot back at start pose with elastic band attached")

    def _subscribe_lowstate(self):  # polls robot state @ 250Hz
        while not self._shutdown_event.is_set():
            start_time = time.time()

            # Step simulation if in simulation mode
            if self.config.is_simulation and self.sim_env is not None:
                self.sim_env.step()
                self._poll_sim_gamepad_buttons()

            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()

                # Capture motor states using jointindex
                for joint in G1_29_JointIndex:
                    lowstate.motor_state[joint].q = msg.motor_state[joint].q
                    lowstate.motor_state[joint].dq = msg.motor_state[joint].dq
                    lowstate.motor_state[joint].tau_est = msg.motor_state[joint].tau_est
                    lowstate.motor_state[joint].temperature = msg.motor_state[joint].temperature

                # Capture IMU state
                lowstate.imu_state.quaternion = list(msg.imu_state.quaternion)
                lowstate.imu_state.gyroscope = list(msg.imu_state.gyroscope)
                lowstate.imu_state.accelerometer = list(msg.imu_state.accelerometer)
                lowstate.imu_state.rpy = list(msg.imu_state.rpy)
                lowstate.imu_state.temperature = msg.imu_state.temperature

                # Capture wireless remote data
                lowstate.wireless_remote = msg.wireless_remote

                # Capture mode_machine
                lowstate.mode_machine = msg.mode_machine

                with self._lowstate_lock:
                    self._lowstate = lowstate

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))  # maintain constant control dt
            time.sleep(sleep_time)

    def publish_lowcmd(
        self,
        action: RobotAction,
        kp: np.ndarray | list[float] | None = None,
        kd: np.ndarray | list[float] | None = None,
        tau: np.ndarray | list[float] | None = None,
    ) -> None:  # writes robot command whenever requested
        with self._lowcmd_lock:
            for motor in G1_29_JointIndex:
                key = f"{motor.name}.q"
                # Slots the body has no motor for keep the zero gains set on connect.
                if key in action and motor.value not in self._invalid_slots:
                    self.msg.motor_cmd[motor.value].q = action[key]
                    self.msg.motor_cmd[motor.value].qd = 0
                    self.msg.motor_cmd[motor.value].kp = (
                        kp[motor.value] if kp is not None else self.kp[motor.value]
                    )
                    self.msg.motor_cmd[motor.value].kd = (
                        kd[motor.value] if kd is not None else self.kd[motor.value]
                    )
                    self.msg.motor_cmd[motor.value].tau = tau[motor.value] if tau is not None else 0.0

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @property
    def _arm_ft(self) -> dict[str, type]:
        return {f"{G1_29_JointArmIndex(motor).name}.q": float for motor in G1_29_JointArmIndex}

    @property
    def _hand_closure(self) -> bool:
        return self.hand_spec is not None and self.config.hand_representation == "closure"

    @property
    def _head_hand_ft(self) -> dict[str, type]:
        """Head joints, then the hand closures or joints."""
        keys = HEAD_KEYS if self.config.head != "none" else ()
        if self.hand_spec is not None:
            keys += HAND_CLOSURE_KEYS if self._hand_closure else self.hand_spec.joint_keys()
        return dict.fromkeys(keys, float)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # With a locomotion controller the policy never commands legs/waist, so they are not recorded.
        motors = self._motors_ft if self.controller is None else self._arm_ft
        return {**motors, **self._head_hand_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.controller is None:
            return {**self._motors_ft, **self._head_hand_ft}
        return {
            **self._arm_ft,
            **self._head_hand_ft,
            **dict.fromkeys(NAV_KEYS, float),
            **self._base_height_features,
        }

    @property
    def _base_height_enabled(self) -> bool:
        return self.config.base_height_action and self.config.controller == "GrootLocomotionController"

    @property
    def _base_height_features(self) -> dict[str, type]:
        return {BASE_HEIGHT_KEY: float} if self._base_height_enabled else {}

    def _controller_loop(self):
        """Background thread that runs controller at policy's control_dt."""
        control_dt = self.controller.control_dt
        logger.info(f"Controller loop starting with control_dt={control_dt} ({1.0 / control_dt:.1f}Hz)")

        loop_count = 0
        last_log_time = time.time()

        while not self._shutdown_event.is_set():
            start_time = time.time()

            with self._lowstate_lock:
                lowstate = self._lowstate

            if lowstate is not None and self.controller is not None:
                loop_count += 1
                if time.time() - last_log_time >= 5.0:  # Log every 5 seconds
                    actual_hz = loop_count / (time.time() - last_log_time)
                    logger.info(
                        f"Controller actual rate: {actual_hz:.1f}Hz (target: {1.0 / control_dt:.1f}Hz)"
                    )
                    loop_count = 0
                    last_log_time = time.time()
                # Read controller input snapshot
                with self._controller_action_lock:
                    controller_input = dict(self.controller_input)

                # Simulation: keep the policy off while the robot hangs on the elastic band (legs limp) and start it
                # from a fresh state when the band is released. Run while hanging, the policy can lock into a
                # sustained leg-kicking oscillation, e.g. after a reset.
                if self._sim_band_attached():
                    if not self._band_was_attached:
                        self._band_was_attached = True
                        self._make_sim_legs_limp()
                    time.sleep(control_dt)
                    continue
                if self._band_was_attached:
                    self._band_was_attached = False
                    if hasattr(self.controller, "reset"):
                        self.controller.reset()

                # Run controller step
                controller_action = self.controller.run_step(controller_input, lowstate)

                # Band re-attached during this step: drop its output so it can't restore the leg gains
                if self._sim_band_attached():
                    continue

                # Write controller output snapshot
                with self._controller_action_lock:
                    self.controller_output = dict(controller_action)

                ctrl_kp = self.controller.kp if hasattr(self.controller, "kp") else None
                ctrl_kd = self.controller.kd if hasattr(self.controller, "kd") else None
                self.publish_lowcmd(controller_action, kp=ctrl_kp, kd=ctrl_kd)

            elapsed = time.time() - start_time
            sleep_time = max(0, control_dt - elapsed)
            time.sleep(sleep_time)

    def _load_calibration(self, fpath: Path | None = None) -> None:
        super()._load_calibration(fpath)
        legacy = [name for name in self.calibration if name in LEGACY_MOTOR_NAMES]
        if legacy:
            self.calibration = {
                LEGACY_MOTOR_NAMES.get(name, name): calib for name, calib in self.calibration.items()
            }
            self._save_calibration(fpath)
            logger.info(f"Renamed {len(legacy)} legacy head/hand motor names in the calibration file")

    def calibrate(self) -> None:
        # Only the head/hand bridge motors need calibrating; the body and Dex hands report joint angles.
        if self.headhand is None:
            return
        print(f"\nCalibrating head/hand motors for {self}")
        self.calibration = self.headhand.calibrate()
        self._save_calibration()

    def configure(self) -> None:
        pass

    def _check_mode_machine(self, mode_machine: int) -> None:
        """23dof only: the robot's mode_machine must match `revision`."""
        if self.config.body != "23dof" or not self.config.check_mode_machine:
            return
        expected = MODE_MACHINE_BY_REVISION[self.config.revision]
        if mode_machine != expected:
            self.disconnect()
            raise ValueError(
                f"mode_machine={mode_machine} does not match revision {self.config.revision!r} "
                f"(expected {expected}); pass --robot.revision=<base|rev_1_0>"
            )

    def _connect_head_hand(self, calibrate: bool) -> None:
        try:
            if self.dex_hand is not None:
                self.dex_hand.connect()
            if self.headhand is not None:
                self.headhand.connect()
        except TimeoutError:
            self.disconnect()
            raise
        if self.headhand is None:
            return
        if calibrate and not self.is_calibrated:
            if self.config.is_simulation:
                logger.info("Simulation mode: writing default head/hand calibration.")
                self.calibration = self.headhand.default_calibration()
                self._save_calibration()
            else:
                self.calibrate()

    def connect(self, calibrate: bool = True) -> None:  # connect to DDS
        # Initialize DDS channel and simulation environment
        if self.config.is_simulation:
            from lerobot.envs import make_env

            self._ChannelFactoryInitialize(0, "lo")
            # The hub env reads the embodiment from these variables (make_env takes no env kwargs).
            embodiment = {
                "UNITREE_G1_MUJOCO_BODY": self.config.body,
                "UNITREE_G1_MUJOCO_END_EFFECTOR": self.config.end_effector,
                "UNITREE_G1_MUJOCO_HEAD": self.config.head,
            }
            with _env_vars(embodiment):
                self._env_wrapper = make_env(self.config.sim_env_repo_id, trust_remote_code=True)
            # Extract the actual gym env from the dict structure
            self.sim_env = self._env_wrapper["hub_env"][0].envs[0]
        else:
            self._ChannelFactoryInitialize(0, config=self.config)

        # Initialize direct motor control interface
        self.lowcmd_publisher = self._ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = self._ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()

        # Start subscribe thread to read robot state
        self.subscribe_thread = threading.Thread(target=self._subscribe_lowstate)
        self.subscribe_thread.start()

        # Connect cameras
        for cam in self._cameras.values():
            if not cam.is_connected:
                cam.connect()

        logger.info(f"Connected {len(self._cameras)} camera(s).")

        # Initialize lowcmd message
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0

        # Wait for first state message to arrive
        lowstate = None
        deadline = time.time() + 10.0
        while lowstate is None:
            with self._lowstate_lock:
                lowstate = self._lowstate
            if lowstate is None:
                if time.time() > deadline:
                    raise TimeoutError("Timed out waiting for robot state (10s)")
                logger.warning("[UnitreeG1] Waiting for robot state...")
                time.sleep(0.01)
        logger.info("[UnitreeG1] Connected to robot.")
        self._check_mode_machine(lowstate.mode_machine)
        self.msg.mode_machine = lowstate.mode_machine

        self.kp = np.array(self.config.kp, dtype=np.float32)
        self.kd = np.array(self.config.kd, dtype=np.float32)

        for joint in G1_29_JointIndex:
            self.msg.motor_cmd[joint].mode = 1
            self.msg.motor_cmd[joint].kp = self.kp[joint.value]
            self.msg.motor_cmd[joint].kd = self.kd[joint.value]
            self.msg.motor_cmd[joint].q = lowstate.motor_state[joint.value].q

        self._connect_head_hand(calibrate)

        # Start controller thread if enabled
        if self.controller is not None:
            self._controller_thread = threading.Thread(target=self._controller_loop, daemon=True)
            self._controller_thread.start()
            fps = int(1.0 / self.controller.control_dt)
            logger.info(f"Controller thread started ({fps}Hz)")

    def _send_zero_torque(self) -> None:
        """Send a zero-gain command to make joints passive before shutting down."""
        try:
            with self._lowstate_lock:
                lowstate = self._lowstate
            if lowstate is None:
                return
            action = {f"{motor.name}.q": lowstate.motor_state[motor.value].q for motor in G1_29_JointIndex}
            zero_gains = np.zeros(29, dtype=np.float32)
            self.publish_lowcmd(action, kp=zero_gains, kd=zero_gains, tau=zero_gains)
            logger.info("Sent zero-torque command for safe shutdown")
        except Exception as e:
            logger.warning(f"Failed to send zero-torque on disconnect: {e}")

    def disconnect(self):
        if self.headhand is not None:
            self.headhand.disconnect()
        if self.dex_hand is not None:
            self.dex_hand.disconnect()
        if self._shutdown_event.is_set():  # already disconnected
            return

        # Put robot in passive mode before stopping threads
        if not self.config.is_simulation:
            self._send_zero_torque()

        # Signal thread to stop and unblock any waits
        self._shutdown_event.set()

        # Wait for subscribe thread to finish
        if self.subscribe_thread is not None:
            self.subscribe_thread.join(timeout=2.0)
            if self.subscribe_thread.is_alive():
                logger.warning("Subscribe thread did not stop cleanly")

        # Wait for controller thread to finish
        if self._controller_thread is not None:
            self._controller_thread.join(timeout=2.0)
            if self._controller_thread.is_alive():
                logger.warning("Controller thread did not stop cleanly")

        # Close simulation environment
        if self.config.is_simulation and self.sim_env is not None:
            try:
                # Force-kill the image publish subprocess first to avoid long waits
                if hasattr(self.sim_env, "simulator") and hasattr(self.sim_env.simulator, "sim_env"):
                    sim_env_inner = self.sim_env.simulator.sim_env
                    if hasattr(sim_env_inner, "image_publish_process"):
                        proc = sim_env_inner.image_publish_process
                        if proc.process and proc.process.is_alive():
                            logger.info("Force-terminating image publish subprocess...")
                            proc.stop_event.set()
                            proc.process.terminate()
                            proc.process.join(timeout=1)
                            if proc.process.is_alive():
                                proc.process.kill()
                self.sim_env.close()
            except Exception as e:
                logger.warning(f"Error closing sim_env: {e}")
            self.sim_env = None
            self._env_wrapper = None

        # Disconnect cameras
        for cam in self._cameras.values():
            cam.disconnect()

        with self._lowstate_lock:
            self._lowstate = None

    def get_observation(self) -> RobotObservation:
        with self._lowstate_lock:
            lowstate = self._lowstate
        if lowstate is None:
            return {}

        obs = {}

        # Motors - q, dq, tau for all joints
        for motor in G1_29_JointIndex:
            name = motor.name
            idx = motor.value
            obs[f"{name}.q"] = lowstate.motor_state[idx].q
            obs[f"{name}.dq"] = lowstate.motor_state[idx].dq
            obs[f"{name}.tau"] = lowstate.motor_state[idx].tau_est

        # IMU - gyroscope
        if lowstate.imu_state.gyroscope:
            obs["imu.gyro.x"] = lowstate.imu_state.gyroscope[0]
            obs["imu.gyro.y"] = lowstate.imu_state.gyroscope[1]
            obs["imu.gyro.z"] = lowstate.imu_state.gyroscope[2]

        # IMU - accelerometer
        if lowstate.imu_state.accelerometer:
            obs["imu.accel.x"] = lowstate.imu_state.accelerometer[0]
            obs["imu.accel.y"] = lowstate.imu_state.accelerometer[1]
            obs["imu.accel.z"] = lowstate.imu_state.accelerometer[2]

        # IMU - quaternion
        if lowstate.imu_state.quaternion:
            obs["imu.quat.w"] = lowstate.imu_state.quaternion[0]
            obs["imu.quat.x"] = lowstate.imu_state.quaternion[1]
            obs["imu.quat.y"] = lowstate.imu_state.quaternion[2]
            obs["imu.quat.z"] = lowstate.imu_state.quaternion[3]

        # IMU - rpy
        if lowstate.imu_state.rpy:
            obs["imu.rpy.roll"] = lowstate.imu_state.rpy[0]
            obs["imu.rpy.pitch"] = lowstate.imu_state.rpy[1]
            obs["imu.rpy.yaw"] = lowstate.imu_state.rpy[2]

        # Wireless remote (raw bytes for teleoperator)
        if lowstate.wireless_remote:
            obs["wireless_remote"] = lowstate.wireless_remote

        # Cameras - read images from ZMQ cameras
        for cam_name, cam in self._cameras.items():
            if getattr(cam, "use_rgb", True):
                obs[cam_name] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs[f"{cam_name}_depth"] = cam.read_latest_depth()

        # Slots the body has no motor for are recorded as 0.0
        for key in self._invalid_body_keys:
            obs[key] = 0.0

        # Head and hand joints, then the hand closures
        if self.headhand is not None:
            obs.update(self.headhand.read(self.calibration))
        if self.dex_hand is not None:
            obs.update(self.dex_hand.read())
        if self._hand_closure:
            for side in HAND_SIDES:
                keys = self.hand_spec.joint_keys(side)
                if all(key in obs for key in keys):
                    obs[hand_closure_key(side)] = self.hand_spec.q_to_closure(
                        side, [obs[key] for key in keys]
                    )

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Command the body, head and hand from one action. Hand closures are expanded to joint targets;
        returns the action with the head/hand targets as sent (clamped)."""
        action = dict(action)
        sent: dict[str, float] = {}

        hand_targets: dict[str, float] = {}
        if self.hand_spec is not None:
            for side in HAND_SIDES:
                key = hand_closure_key(side)
                if self._hand_closure and key in action:
                    closure = min(max(float(action.pop(key)), 0.0), 1.0)
                    sent[key] = closure
                    action.update(
                        zip(
                            self.hand_spec.joint_keys(side),
                            self.hand_spec.closure_to_q(side, closure),
                            strict=True,
                        )
                    )
            hand_targets = {key: action.pop(key) for key in self.hand_spec.joint_keys() if key in action}
        head_targets = {key: action.pop(key) for key in HEAD_KEYS if key in action}

        body_sent = self._send_body_action(action)

        if self.headhand is not None:
            bridge_targets = {**head_targets, **(hand_targets if self.dex_hand is None else {})}
            sent.update(self.headhand.write(bridge_targets, self.calibration))
        if self.dex_hand is not None and hand_targets:
            self.dex_hand.write(hand_targets)
            sent.update(hand_targets)
        return {**body_sent, **sent}

    def _send_body_action(self, action: RobotAction) -> RobotAction:
        action_to_publish = action
        if self.controller is not None:
            # Controller thread owns legs/waist. Here we only update joystick inputs
            # and publish arm targets from the teleoperator.
            self._update_controller_action(action)
            arm_prefixes = tuple(j.name for j in G1_29_JointArmIndex)
            action_to_publish = {
                key: value
                for key, value in action.items()
                if key.endswith(".q") and key.startswith(arm_prefixes)
            }

        tau = None
        if self.config.gravity_compensation and self.arm_ik is not None:
            tau = np.zeros(29, dtype=np.float32)
            action_np = np.array(
                [
                    action_to_publish.get(f"{joint.name}.q", self.msg.motor_cmd[joint.value].q)
                    for joint in G1_29_JointArmIndex
                ],
                dtype=np.float32,
            )
            arm_tau = self.arm_ik.solve_tau(action_np)
            arm_start_idx = G1_29_JointArmIndex.kLeftShoulderPitch.value
            for joint in G1_29_JointArmIndex:
                local_idx = joint.value - arm_start_idx
                tau[joint.value] = arm_tau[local_idx]

        self.publish_lowcmd(action_to_publish, tau=tau)
        return action

    def _update_controller_action(self, action: RobotAction) -> None:
        """Update controller input state from incoming teleop action."""
        with self._controller_action_lock:
            for key in REMOTE_KEYS:
                if key in action:
                    self.controller_input[key] = action[key]
            for key in NAV_KEYS:
                if key in action:
                    self.controller_input[key] = action[key]
            if self._base_height_enabled and BASE_HEIGHT_KEY in action:
                self.controller_input[BASE_HEIGHT_KEY] = action[BASE_HEIGHT_KEY]

    @property
    def is_calibrated(self) -> bool:
        return self.headhand is None or self.headhand.is_calibrated(self.calibration)

    @property
    def is_connected(self) -> bool:
        with self._lowstate_lock:
            return self._lowstate is not None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Joint positions for all 29 joints."""
        return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

    @property
    def cameras(self) -> dict:
        return self._cameras

    def reset(
        self,
        control_dt: float | None = None,
        default_positions: list[float] | None = None,
    ) -> None:  # move robot to default position
        if control_dt is None:
            control_dt = self.config.control_dt
        if default_positions is None:
            default_positions = np.array(self.config.default_positions, dtype=np.float32)

        if self.config.is_simulation and self.sim_env is not None:
            self.sim_env.reset()
            self.publish_lowcmd(
                {f"{motor.name}.q": float(default_positions[motor.value]) for motor in G1_29_JointIndex}
            )
        else:
            total_time = 3.0
            num_steps = int(total_time / control_dt)

            # get current state
            obs = self.get_observation()

            # record current positions
            init_dof_pos = np.zeros(29, dtype=np.float32)
            for motor in G1_29_JointIndex:
                init_dof_pos[motor.value] = obs[f"{motor.name}.q"]

            # Interpolate to default position
            for step in range(num_steps):
                start_time = time.time()

                alpha = step / num_steps
                action_dict = {}
                for motor in G1_29_JointIndex:
                    target_pos = default_positions[motor.value]
                    interp_pos = init_dof_pos[motor.value] * (1 - alpha) + target_pos * alpha
                    action_dict[f"{motor.name}.q"] = float(interp_pos)

                self.send_action(action_dict)

                # Maintain constant control rate
                elapsed = time.time() - start_time
                sleep_time = max(0, control_dt - elapsed)
                time.sleep(sleep_time)

        # Reset controller internal state (gait phase, obs history, etc.)
        if self.controller is not None and hasattr(self.controller, "reset"):
            self.controller.reset()

        self._reset_head_hand(control_dt)

        logger.info("Reached default position")

    def _reset_head_hand(self, control_dt: float, total_time: float = 2.0) -> None:
        """Ramp the head and hand joints to their default positions."""
        keys = list(HEAD_KEYS) if self.config.head != "none" else []
        keys += list(self.hand_spec.joint_keys()) if self.hand_spec is not None else []
        if not keys:
            return
        targets = [*self.config.head_default_positions, *self.config.hand_default_positions]
        obs = self.get_observation()
        start = [obs.get(key, target) for key, target in zip(keys, targets, strict=True)]
        num_steps = max(1, int(total_time / control_dt))
        for step in range(num_steps):
            step_start = time.time()
            alpha = step / num_steps
            self.send_action(
                {key: s * (1 - alpha) + t * alpha for key, s, t in zip(keys, start, targets, strict=True)}
            )
            time.sleep(max(0, control_dt - (time.time() - step_start)))
