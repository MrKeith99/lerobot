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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from .end_effectors import AMAZING_HAND_MOTORS, END_EFFECTORS, HAND_REPRESENTATIONS, HAND_SIDES, HAND_SPECS
from .g1_utils import BODIES, G1_LEG_SLOTS, REVISIONS, invalid_sdk_slots
from .headhand_zmq import HEADHAND_CMD_PORT, HEADHAND_STATE_PORT
from .heads import DEFAULT_HEAD_Q, HEAD_MOTORS, HEADS

_GAINS: dict[str, dict[str, list[float]]] = {
    "left_leg": {
        "kp": [150, 150, 150, 300, 40, 40],
        "kd": [2, 2, 2, 4, 2, 2],
    },  # pitch, roll, yaw, knee, ankle_pitch, ankle_roll
    "right_leg": {"kp": [150, 150, 150, 300, 40, 40], "kd": [2, 2, 2, 4, 2, 2]},
    "waist": {"kp": [250, 250, 250], "kd": [5, 5, 5]},  # yaw, roll, pitch
    "left_arm": {"kp": [50, 50, 80, 80], "kd": [3, 3, 3, 3]},  # shoulder_pitch/roll/yaw, elbow
    "left_wrist": {"kp": [40, 40, 40], "kd": [1.5, 1.5, 1.5]},  # roll, pitch, yaw
    "right_arm": {"kp": [50, 50, 80, 80], "kd": [3, 3, 3, 3]},
    "right_wrist": {"kp": [40, 40, 40], "kd": [1.5, 1.5, 1.5]},
}


def _build_gains() -> tuple[list[float], list[float]]:
    """Build kp and kd lists from body-part groupings."""
    kp = [v for g in _GAINS.values() for v in g["kp"]]
    kd = [v for g in _GAINS.values() for v in g["kd"]]
    return kp, kd


_DEFAULT_KP, _DEFAULT_KD = _build_gains()


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
    # Embodiment, mirroring the MuJoCo sim's BODY / END_EFFECTOR / HEAD options; any combination:
    # body "29dof" or "23dof" (no waist roll/pitch, no wrist pitch/yaw);
    # end_effector "rubber_hand" (stock passive hand), "none" (bare wrist), "dex1" (gripper),
    # "dex3" (hand) or "amazing_hand"; head "none" or "d455_pan_tilt" (Dynamixel pan/tilt D455).
    body: str = "29dof"
    end_effector: str = "rubber_hand"
    head: str = "none"

    # 23dof only: hardware revision, checked against the robot's reported mode_machine.
    revision: str = "rev_1_0"
    check_mode_machine: bool = True

    # "closure": one value per hand in [0, 1] (0 = open, 1 = closed); "per_motor": every hand joint.
    hand_representation: str = "closure"

    kp: list[float] = field(default_factory=lambda: _DEFAULT_KP.copy())
    kd: list[float] = field(default_factory=lambda: _DEFAULT_KD.copy())

    # Default joint positions
    default_positions: list[float] = field(default_factory=lambda: [0.0] * 29)

    # Control loop timestep
    control_dt: float = 1.0 / 250.0  # 250Hz

    # Launch mujoco simulation
    is_simulation: bool = True

    # HF Hub repo id passed to `make_env(..., trust_remote_code=True)` when is_simulation is True
    sim_env_repo_id: str = "k-valentin/unitree-g1-mujoco"

    # Socket config for ZMQ bridge
    robot_ip: str = "192.168.123.164"  # default G1 IP

    # Cameras (ZMQ-based remote cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Compensates for gravity on the unitree's arms using the arm ik solver
    gravity_compensation: bool = False

    # Lower-body controller class name, e.g. "GrootLocomotionController" or
    # "HolosomaLocomotionController". None disables it.
    controller: str | None = None

    # GrootLocomotionController only: record and command the base height as an absolute action,
    # `kBaseHeight.cmd` (m), instead of integrating remote R1/R2 inside the controller.
    base_height_action: bool = True

    # Simulation only: gamepad (pygame) button index that toggles the MuJoCo elastic band, same as
    # pressing "9" in the viewer. 10 = PS button on a DualShock 4. None disables it.
    sim_band_toggle_button: int | None = 10

    # Simulation only: gamepad (pygame) button index that resets the robot to its start pose with the
    # elastic band attached. 9 = Options button on a DualShock 4. None disables it.
    sim_reset_button: int | None = 9

    # Without a controller: zero the leg gains so the legs hang passive (e.g. on a gantry).
    freeze_legs: bool = False

    # Head/hand ZMQ bridge (d455_pan_tilt head, amazing_hand). Defaults to `robot_ip` on hardware, or
    # 127.0.0.1 in simulation (the sim bridge always binds to localhost); set explicitly to override.
    headhand_ip: str | None = None
    headhand_state_port: int = HEADHAND_STATE_PORT
    headhand_cmd_port: int = HEADHAND_CMD_PORT
    headhand_timeout_s: float = 5.0
    headhand_stale_warn_s: float = 0.5

    # Reset targets of the head (yaw, pitch) and hands (left then right joints). None = centered / open.
    head_default_positions: list[float] | None = None
    hand_default_positions: list[float] | None = None

    def __post_init__(self):
        super().__post_init__()
        for name, value, choices in (
            ("body", self.body, BODIES),
            ("end_effector", self.end_effector, END_EFFECTORS),
            ("head", self.head, HEADS),
            ("revision", self.revision, REVISIONS),
            ("hand_representation", self.hand_representation, HAND_REPRESENTATIONS),
        ):
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}, got {value!r}")

        if not (len(self.kp) == len(self.kd) == len(self.default_positions) == 29):
            raise ValueError("kp, kd and default_positions must all have length 29")
        if self.head_default_positions is None:
            self.head_default_positions = list(DEFAULT_HEAD_Q) if self.head != "none" else []
        expected_head = len(DEFAULT_HEAD_Q) if self.head != "none" else 0
        if len(self.head_default_positions) != expected_head:
            raise ValueError(f"head_default_positions must have length {expected_head}")
        spec = HAND_SPECS.get(self.end_effector)
        if self.hand_default_positions is None:
            self.hand_default_positions = (
                [q for side in HAND_SIDES for q in spec.open_q[side]] if spec else []
            )
        expected_hand = len(spec.joint_keys()) if spec else 0
        if len(self.hand_default_positions) != expected_hand:
            raise ValueError(f"hand_default_positions must have length {expected_hand}")

        for i in invalid_sdk_slots(self.body):
            self.kp[i] = 0.0
            self.kd[i] = 0.0
        if self.freeze_legs and self.controller is None:
            for i in G1_LEG_SLOTS:
                self.kp[i] = 0.0
                self.kd[i] = 0.0

    @property
    def robot_type(self) -> str:
        """Embodiment name, used as the dataset `robot_type` and calibration id, e.g.
        `unitree_g1_23dof_amazing_hand_d455_pan_tilt`. The stock rubber hand and fixed head are
        left out; a bare wrist is `no_hand`."""
        parts = ["unitree_g1", self.body]
        if self.end_effector != "rubber_hand":
            parts.append("no_hand" if self.end_effector == "none" else self.end_effector)
        if self.head != "none":
            parts.append(self.head)
        return "_".join(parts)

    @property
    def headhand_motors(self) -> dict[str, tuple[int, str]]:
        """Motors behind the head/hand ZMQ bridge, {name: (ID, model)}; empty if there are none."""
        motors = dict(HEAD_MOTORS) if self.head == "d455_pan_tilt" else {}
        if self.end_effector == "amazing_hand":
            motors.update(AMAZING_HAND_MOTORS)
        return motors
