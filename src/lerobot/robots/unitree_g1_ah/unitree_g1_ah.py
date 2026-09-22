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

"""UnitreeG1Ah robot: `UnitreeG1` body plus a ZMQ-bridged Dynamixel head and AmazingHand hands."""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from functools import cached_property

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

from .config_unitree_g1_ah import UnitreeG1AhConfig
from .g1_ah_devices import (
    TICK_RANGE,
    TICKS_PER_RAD,
    clamp_rad,
    default_calibration,
    rad_to_ticks,
    ticks_to_rad,
)
from .g1_ah_joints import (
    ALL_ACTION_KEYS,
    ARM_MODE_ACTION_KEYS,
    G1_23_INVALID_SDK_SLOTS,
    HAND_LIMIT_RAD,
    HEAD_HAND_KEYS,
    HEAD_HAND_MOTORS,
    HEAD_LIMITS_RAD,
    HEAD_MOTORS,
    INVALID_BODY_KEYS,
    MIDDLE_POS_DEG,
    MODE_MACHINE_BY_REVISION,
    ROBOT_TYPE_BASE,
    hand_motor_names,
    key_to_motor_name,
)
from .g1_ah_zmq import HeadHandZmqClient

logger = logging.getLogger(__name__)

_HEAD_HAND_KEY_SET = frozenset(HEAD_HAND_KEYS)


def _clamp_to_model_range(model: str, value: int) -> int:
    tick_min, tick_max = TICK_RANGE[model]
    return max(tick_min, min(value, tick_max))


class UnitreeG1Ah(UnitreeG1):
    config_class = UnitreeG1AhConfig
    name = ROBOT_TYPE_BASE

    def __init__(self, config: UnitreeG1AhConfig):
        self.name = config.robot_type_name
        super().__init__(config)
        headhand_ip = config.headhand_ip or ("127.0.0.1" if config.is_simulation else config.robot_ip)
        self.headhand = HeadHandZmqClient(headhand_ip, config.headhand_state_port, config.headhand_cmd_port)
        self._headhand_ticks: dict[str, int] = {}
        self._headhand_stale_logged = False
        self._headhand_warned_names: set[str] = set()
        self._invalid_slots = np.array(G1_23_INVALID_SDK_SLOTS)
        # send_action (main thread) and the locomotion controller thread both publish lowcmd;
        # CycloneDDS delivers rt/lowcmd inline to in-process readers, so writes must not overlap.
        self._lowcmd_lock = threading.Lock()

    @property
    def _motors_ft(self) -> dict[str, type]:
        return dict.fromkeys(ALL_ACTION_KEYS, float)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.controller is None:
            return dict.fromkeys(ALL_ACTION_KEYS, float)
        return dict.fromkeys(ARM_MODE_ACTION_KEYS, float)

    def publish_lowcmd(
        self,
        action: RobotAction,
        kp: np.ndarray | list[float] | None = None,
        kd: np.ndarray | list[float] | None = None,
        tau: np.ndarray | list[float] | None = None,
    ) -> None:
        kp_arr = np.array(kp if kp is not None else self.kp, dtype=np.float32).copy()
        kd_arr = np.array(kd if kd is not None else self.kd, dtype=np.float32).copy()
        tau_arr = None if tau is None else np.array(tau, dtype=np.float32).copy()
        kp_arr[self._invalid_slots] = 0.0
        kd_arr[self._invalid_slots] = 0.0
        if tau_arr is not None:
            tau_arr[self._invalid_slots] = 0.0

        body_action = {k: v for k, v in action.items() if k not in INVALID_BODY_KEYS}
        with self._lowcmd_lock:
            super().publish_lowcmd(body_action, kp=kp_arr, kd=kd_arr, tau=tau_arr)

    def _check_mode_machine(self) -> None:
        if not self.config.check_mode_machine:
            return
        with self._lowstate_lock:
            got = self._lowstate.mode_machine
        expected = MODE_MACHINE_BY_REVISION[self.config.revision]
        if got != expected:
            self.disconnect()
            raise ValueError(
                f"mode_machine={got} does not match revision {self.config.revision!r} "
                f"(expected {expected}); pass --robot.revision=<base|rev_1_0>"
            )

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=False)
        self._check_mode_machine()

        self.headhand.connect()
        deadline = time.time() + self.config.headhand_timeout_s
        while time.time() < deadline:
            data = self.headhand.read_latest()
            if data is not None and data.get("ticks"):
                self._headhand_ticks = dict(data["ticks"])
                break
            time.sleep(0.01)
        else:
            self.disconnect()
            raise TimeoutError(
                f"Timed out waiting for head/hand state on port {self.config.headhand_state_port}"
            )

        if calibrate and not self.is_calibrated:
            if self.config.is_simulation:
                logger.info("Simulation mode: writing default head/hand calibration.")
                self.calibration = {name: default_calibration(name) for name in HEAD_HAND_MOTORS}
                self._save_calibration()
            else:
                self.calibrate()

        self.configure()

    @property
    def is_calibrated(self) -> bool:
        return all(name in self.calibration for name in HEAD_HAND_MOTORS)

    def _record_range(self, seconds: float | None = None) -> dict[str, tuple[int, int]]:
        """Sample `read_latest()` ticks until Enter is pressed, tracking per-motor min/max."""
        mins: dict[str, int] = {}
        maxs: dict[str, int] = {}
        stop_event = threading.Event()

        def _wait_enter() -> None:
            input()
            stop_event.set()

        waiter = threading.Thread(target=_wait_enter, daemon=True)
        waiter.start()
        deadline = None if seconds is None else time.time() + seconds
        while not stop_event.is_set() and (deadline is None or time.time() < deadline):
            data = self.headhand.read_latest()
            if data and data.get("ticks"):
                for name, tick in data["ticks"].items():
                    mins[name] = tick if name not in mins else min(mins[name], tick)
                    maxs[name] = tick if name not in maxs else max(maxs[name], tick)
            time.sleep(0.01)
        waiter.join(timeout=0.1)
        return {name: (mins[name], maxs[name]) for name in mins}

    def _calibrate_head(self, head_range: dict[str, tuple[int, int]], zero_ticks, direction_ticks) -> dict:
        calibration: dict[str, MotorCalibration] = {}
        for name, (motor_id, model) in HEAD_MOTORS.items():
            zero = zero_ticks.get(name, default_calibration(name).homing_offset)
            drive_mode = 0
            if name in direction_ticks and name in zero_ticks:
                drive_mode = 0 if direction_ticks[name] >= zero_ticks[name] else 1
            low_rad, high_rad = HEAD_LIMITS_RAD[name]
            sign = -1.0 if drive_mode else 1.0
            tick_low = round(sign * low_rad * TICKS_PER_RAD[model]) + zero
            tick_high = round(sign * high_rad * TICKS_PER_RAD[model]) + zero
            range_min, range_max = min(tick_low, tick_high), max(tick_low, tick_high)
            if name in head_range:
                sample_min, sample_max = head_range[name]
                range_min = max(range_min, sample_min)
                range_max = min(range_max, sample_max)
            range_min = _clamp_to_model_range(model, range_min)
            range_max = _clamp_to_model_range(model, range_max)
            calibration[name] = MotorCalibration(
                id=motor_id,
                drive_mode=drive_mode,
                homing_offset=zero,
                range_min=range_min,
                range_max=range_max,
            )
        return calibration

    def _calibrate_hands_interactive(self) -> dict:
        input("Move both hands to a neutral pose, then press Enter")
        neutral_data = self.headhand.read_latest()
        neutral_ticks = dict(neutral_data["ticks"]) if neutral_data and neutral_data.get("ticks") else {}
        print("Sweep both hands through their full range, then press Enter")
        hand_range = self._record_range()

        calibration: dict[str, MotorCalibration] = {}
        for side in MIDDLE_POS_DEG:
            for motor_name in hand_motor_names(side):
                motor_id, model = HEAD_HAND_MOTORS[motor_name]
                zero = neutral_ticks.get(motor_name, default_calibration(motor_name).homing_offset)
                span = round(HAND_LIMIT_RAD * TICKS_PER_RAD[model])
                range_min, range_max = zero - span, zero + span
                if motor_name in hand_range:
                    sample_min, sample_max = hand_range[motor_name]
                    range_min = max(range_min, sample_min)
                    range_max = min(range_max, sample_max)
                range_min = _clamp_to_model_range(model, range_min)
                range_max = _clamp_to_model_range(model, range_max)
                calibration[motor_name] = MotorCalibration(
                    id=motor_id,
                    drive_mode=0,
                    homing_offset=zero,
                    range_min=range_min,
                    range_max=range_max,
                )
        return calibration

    def _calibrate_hands_from_lab_offsets(self) -> dict:
        calibration: dict[str, MotorCalibration] = {}
        for side, deg_offsets in MIDDLE_POS_DEG.items():
            for deg, motor_name in zip(deg_offsets, hand_motor_names(side), strict=True):
                motor_id, model = HEAD_HAND_MOTORS[motor_name]
                zero = 512 + round(deg * TICKS_PER_RAD[model] * math.pi / 180.0)
                span = round(HAND_LIMIT_RAD * TICKS_PER_RAD[model])
                range_min = _clamp_to_model_range(model, zero - span)
                range_max = _clamp_to_model_range(model, zero + span)
                calibration[motor_name] = MotorCalibration(
                    id=motor_id,
                    drive_mode=0,
                    homing_offset=zero,
                    range_min=range_min,
                    range_max=range_max,
                )
        return calibration

    def calibrate(self) -> None:
        print(f"\nCalibrating UnitreeG1Ah head/hand motors for {self}")
        self.headhand.send(torque=False)

        input("Center the head (camera forward, level), then press Enter")
        zero_data = self.headhand.read_latest()
        zero_ticks = dict(zero_data["ticks"]) if zero_data and zero_data.get("ticks") else {}

        input("Turn the head to the robot's LEFT and tilt UP slightly, then press Enter")
        direction_data = self.headhand.read_latest()
        direction_ticks = (
            dict(direction_data["ticks"]) if direction_data and direction_data.get("ticks") else {}
        )

        print("Sweep pan and tilt to both mechanical stops, then press Enter")
        head_range = self._record_range()

        calibration = self._calibrate_head(head_range, zero_ticks, direction_ticks)

        hand_mode = input("Press Enter to import the lab middle_pos offsets, or type 'i' for interactive: ")
        if hand_mode.strip().lower() == "i":
            calibration.update(self._calibrate_hands_interactive())
        else:
            calibration.update(self._calibrate_hands_from_lab_offsets())

        self.calibration = calibration
        self._save_calibration()
        self.headhand.send(torque=True)

    def configure(self) -> None:
        """No-op: the head/hand ZMQ bridge server configures its own motors on connect."""

    def get_observation(self) -> RobotObservation:
        obs = super().get_observation()

        data = self.headhand.read_latest()
        if data and data.get("ticks"):
            self._headhand_ticks.update(data["ticks"])

        age_s = self.headhand.age_s
        if age_s is not None and age_s > self.config.headhand_stale_warn_s:
            if not self._headhand_stale_logged:
                logger.warning(f"Head/hand state is stale ({age_s:.2f}s old)")
                self._headhand_stale_logged = True
        else:
            self._headhand_stale_logged = False

        for name, (_motor_id, model) in HEAD_HAND_MOTORS.items():
            if name in self._headhand_ticks and name in self.calibration:
                ticks = self._headhand_ticks[name]
                obs[f"{name}.q"] = ticks_to_rad(model, ticks, self.calibration[name])

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        body_action = {k: v for k, v in action.items() if k not in _HEAD_HAND_KEY_SET}
        sent = super().send_action(body_action)

        goals: dict[str, int] = {}
        for key in HEAD_HAND_KEYS:
            if key not in action:
                continue
            name = key_to_motor_name(key)
            if name not in self.calibration:
                if name not in self._headhand_warned_names:
                    logger.warning(f"No calibration for {name!r}; skipping in send_action")
                    self._headhand_warned_names.add(name)
                continue
            _motor_id, model = HEAD_HAND_MOTORS[name]
            rad = clamp_rad(name, float(action[key]))
            goals[name] = rad_to_ticks(model, rad, self.calibration[name])
            sent[key] = rad

        if goals:
            self.headhand.send(goal_ticks=goals)

        return sent

    def reset(
        self,
        control_dt: float | None = None,
        default_positions: list[float] | None = None,
    ) -> None:
        super().reset(control_dt, default_positions)

        dt = control_dt if control_dt is not None else self.control_dt
        total_time = 2.0
        num_steps = max(1, int(total_time / dt))

        obs = self.get_observation()
        head_hand_keys = list(HEAD_HAND_KEYS)
        targets = [*self.config.head_default_positions, *self.config.hand_default_positions]
        start = [obs.get(key, 0.0) for key in head_hand_keys]

        for step in range(num_steps):
            step_start = time.time()
            alpha = step / num_steps
            action = {
                key: start[i] * (1 - alpha) + targets[i] * alpha for i, key in enumerate(head_hand_keys)
            }
            self.send_action(action)
            elapsed = time.time() - step_start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)

    def disconnect(self) -> None:
        with contextlib.suppress(Exception):
            self.headhand.disconnect()
        if not self.is_connected:
            return
        super().disconnect()
        with self._lowstate_lock:
            self._lowstate = None
