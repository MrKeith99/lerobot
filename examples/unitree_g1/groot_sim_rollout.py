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

"""Closed-loop rollout of a LeRobot GR00T N1.7 checkpoint on the simulated Unitree G1 (MuJoCo).

`lerobot-rollout` only routes `.pos`/`.vel` features and `GrootPolicy.select_action` does not support
relative-action policies, so this script drives the robot directly. It builds the simulated G1
(23dof, AmazingHand, pan/tilt D455, GR00T locomotion controller, a world such as ``pick_cylinder`` or
``table_room_1x3450_short`` with the ``head_camera`` and ``table_view`` ZMQ cameras), waits until the elastic band is
released, then every ``--execution-horizon`` ticks runs preprocessor -> ``predict_action_chunk`` -> postprocessor on the
current observation and sends the resulting absolute actions one per tick at ``--fps``. Inference runs in a background thread
(`lerobot.utils.chunk_scheduler`): the running chunk keeps playing while the next one is computed, and the new one starts at
the action for the current tick (the ones that went stale meanwhile are skipped), so the 30 Hz loop never stalls.
``--sync-inference`` restores the blocking loop of the evals before 2026-11-03 (each inference stalls the loop ~130 ms, then
the missed ticks are sent back to back). One warm-up inference runs before the rollout, so the first chunk does not stall. The height of the world's free
object (``--object-body``, read from the MuJoCo state) is tracked to report the maximum / final lift. A run succeeds when
the object stays more than ``--success-lift`` above its start for ``--success-hold-s`` in a row (lifted and held, so a knock
or bounce does not count). An mp4 with both cameras side by side is saved.

The result json also has an ``outcome`` (success / knocked_off / toppled / pushed / touched / untouched) computed from the
object's trajectory and its contacts with the robot. ``--trace-npz`` saves the per-tick data behind it for offline analysis:
state, action, chunk id, object / base / hand poses, hand-object distance, contact counts, loop timing, and per chunk the
inference time and the full predicted chunk (arrays plus a ``meta`` JSON string with the args, seed, layout and revisions).

Run it headless (no display) with the same environment as the XR sim recording:

```shell
export CYCLONEDDS_HOME=~/Documents/g1_dev/cyclonedds/install LD_LIBRARY_PATH=~/miniconda3/envs/lerobot/lib \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl UNITREE_G1_MUJOCO_ONSCREEN=0 SDL_VIDEODRIVER=dummy
unset DISPLAY
python examples/unitree_g1/groot_sim_rollout.py \
    --policy-path outputs/train/my_run/checkpoints/last/pretrained_model \
    --world pick_cylinder --duration 40 --execution-horizon 16 \
    --seed 3 --stop-after-success-s 3 --result-json outputs/eval/seed_03.json
# table_room: object pringles_can, any heading, task from the training dataset, per-tick trace
python examples/unitree_g1/groot_sim_rollout.py \
    --policy-path outputs/train/groot_100ep_chips/checkpoints/last/pretrained_model \
    --world table_room_1x3450_short --randomize-yaw 180 --seed 3 --stop-after-success-s 3 \
    --result-json outputs/eval/seed_03.json --trace-npz outputs/eval/seed_03.npz
```

The defaults match the XR sim recording (`guides/run_xr_sim_record.sh`): world jitter 0.08, robot start jitter (0.15 m, 15 deg),
object re-placed 3 s after the band release; the start offset is (0, 0, 0) for ``table_room_*`` worlds (the spawn is already
at the table) and -1.3 m back (``--start-offset -1.3 0 0``) otherwise. ``--randomize-yaw`` is only passed to the sim when given
(its own default is 15 deg), ``--task`` defaults to the training dataset's task.
``--seed`` makes the policy noise, the start pose and the object jitter (position and yaw, both drawn from the sim's `world_rng`)
reproducible (same layout for every checkpoint); ``--result-json`` writes one machine-readable result per run
(see `guides/run_groot_eval.sh`).
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots import make_robot_from_config
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.utils.chunk_scheduler import ChunkScheduler
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger("groot_sim_rollout")

HEAD_CAM = "head_camera"
TABLE_CAM = "table_view"
IMAGE_PREFIX = "observation.images."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-path", required=True, help="Path to a checkpoint's `pretrained_model` dir.")
    parser.add_argument(
        "--task", default=None, help="Language task; default: the training dataset's (first) task."
    )
    parser.add_argument("--world", default="pick_cylinder", help="Sim world name.")
    parser.add_argument("--randomize", type=float, default=0.08, help="World object xy jitter in metres.")
    parser.add_argument(
        "--randomize-yaw",
        type=float,
        default=None,
        help="World object yaw jitter, +- degrees (0..180); default: the sim's own (15).",
    )
    parser.add_argument(
        "--object-body",
        default=None,
        help="MuJoCo body of the object to lift; default: `cylinder` if the world has it, else its single free object.",
    )
    parser.add_argument(
        "--start-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "YAW"),
        help="Robot start relative to the spawn (m, m, deg); default: 0 0 0 for table_room_* worlds (the spawn is "
        "already at the table), else 1.3 m back so it has to walk up to the table.",
    )
    parser.add_argument(
        "--start-randomize",
        type=float,
        nargs=2,
        default=[0.15, 15.0],
        metavar=("XY", "YAW"),
        help="Uniform start jitter added to --start-offset (m, deg).",
    )
    parser.add_argument(
        "--replace-after-release-s",
        type=float,
        default=3.0,
        help="Seconds after the band release to re-place the robot.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed torch / numpy and the sim start + object jitter."
    )
    parser.add_argument(
        "--no-collision-left-arm",
        action="store_true",
        help="Diagnostic: make the left arm and hand (left_shoulder_pitch_link and below) collide with nothing, so a low "
        "left arm can't catch the table edge and hold the robot back.",
    )
    parser.add_argument("--duration", type=float, default=40.0, help="Rollout length in seconds.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--execution-horizon", type=int, default=16, help="Actions executed per predicted chunk."
    )
    parser.add_argument(
        "--sync-inference",
        action="store_true",
        help="Block the control loop during inference (the behaviour before 2026-11-03) instead of running it in a "
        "background thread.",
    )
    parser.add_argument(
        "--band-release-s", type=float, default=2.0, help="Seconds after connect to release the band."
    )
    parser.add_argument(
        "--success-lift", type=float, default=0.05, help="Object lift (m) counting as success."
    )
    parser.add_argument(
        "--success-hold-s",
        type=float,
        default=1.0,
        help="Seconds the lift must stay above --success-lift in a row.",
    )
    parser.add_argument(
        "--stop-after-success-s",
        type=float,
        default=None,
        help="End the rollout this long after success (lift held for --success-hold-s).",
    )
    parser.add_argument(
        "--result-json", default=None, help="Write the run's result (one JSON object) to this path."
    )
    parser.add_argument(
        "--trace-npz",
        default=None,
        help="Write the per-tick / per-chunk trace for offline analysis (compressed npz, see the module docstring).",
    )
    parser.add_argument(
        "--video-out", default=None, help="mp4 path (default outputs/rollouts/<timestamp>.mp4)."
    )
    parser.add_argument(
        "--no-gravity-compensation", action="store_true", help="Disable the arm gravity feed-forward."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ready-timeout", type=float, default=60.0, help="Max seconds to wait for the band release."
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Training dataset (for state/action names); default: from train_config.json.",
    )
    args = parser.parse_args()
    if args.start_offset is None:
        args.start_offset = [0.0, 0.0, 0.0] if "table_room_" in args.world else [-1.3, 0.0, 0.0]
    return args


def load_feature_names(policy_path: str, repo_id: str | None) -> tuple[list[str], list[str], str | None]:
    """State and action names in training-dataset order, and the dataset's first task (None if it has none)."""
    from lerobot.datasets import LeRobotDatasetMetadata

    if repo_id is None:
        repo_id = json.loads((Path(policy_path) / "train_config.json").read_text())["dataset"]["repo_id"]
    meta = LeRobotDatasetMetadata(repo_id)
    features = meta.features
    task = str(meta.tasks.index[0]) if meta.tasks is not None and len(meta.tasks) else None
    return list(features[OBS_STATE]["names"]), list(features[ACTION]["names"]), task


def build_robot(args: argparse.Namespace):
    def cam(name: str) -> ZMQCameraConfig:
        return ZMQCameraConfig(
            server_address="127.0.0.1", port=5555, camera_name=name, width=640, height=480, fps=30, warmup_s=5
        )

    config = UnitreeG1Config(
        id="sim",
        body="23dof",
        end_effector="amazing_hand",
        head_mount="pan_tilt",
        head_sensor="d455",
        is_simulation=True,
        controller="GrootLocomotionController",
        sim_world=args.world,
        sim_world_randomize=args.randomize,
        **({} if args.randomize_yaw is None else {"sim_world_randomize_yaw": args.randomize_yaw}),
        sim_world_grasp="physics",
        sim_start_offset=tuple(args.start_offset),
        sim_start_randomize=tuple(args.start_randomize),
        sim_replace_after_release_s=args.replace_after_release_s,
        gravity_compensation=not args.no_gravity_compensation,
        sim_band_release_s=args.band_release_s,
        cameras={HEAD_CAM: cam(HEAD_CAM), TABLE_CAM: cam(TABLE_CAM)},
    )
    return make_robot_from_config(config)


class VideoWriter:
    """Side-by-side mp4 writer on top of PyAV."""

    def __init__(self, path: Path, fps: int):
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), "w")
        try:
            self.stream = self.container.add_stream("libx264", rate=fps)
            self.stream.options = {"crf": "23", "preset": "veryfast"}
        except Exception:  # noqa: BLE001 - encoder not built in
            self.stream = self.container.add_stream("mpeg4", rate=fps)
        self.stream.pix_fmt = "yuv420p"
        self.size = None

    def write(self, frame: np.ndarray) -> None:
        import av

        if self.size is None:
            h, w = frame.shape[:2]
            self.size = (w - w % 2, h - h % 2)
            self.stream.width, self.stream.height = self.size
        frame = np.ascontiguousarray(frame[: self.size[1], : self.size[0]])
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            self.container.mux(packet)

    def close(self) -> None:
        if self.size is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
        self.container.close()


def tilt_deg(quat: np.ndarray) -> np.ndarray:
    """Angle (deg) between the body's z axis and the world z axis, from wxyz quaternion(s) of shape (..., 4)."""
    quat = np.asarray(quat, dtype=float)
    cos = 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)  # z component of R @ [0, 0, 1]
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def classify_outcome(
    success: bool,
    lift: np.ndarray,
    object_pos: np.ndarray,
    tilt: np.ndarray,
    n_contacts: np.ndarray,
    hand_dist: np.ndarray,
    fps: int,
    success_lift: float,
) -> dict:
    """Outcome of a run from per-tick object lift (m), position (N, 3), tilt (deg), robot-object contact count, hand-object distance.

    success > knocked_off (ends > 0.1 m below its start, i.e. below the table top) > toppled (final tilt > 45 deg)
    > pushed (upright, moved > 3 cm in xy, not lifted) > touched (some robot-object contact) > untouched.
    """
    lift, tilt = np.asarray(lift), np.asarray(tilt)
    xy_disp = float(np.linalg.norm(np.asarray(object_pos)[-1, :2] - np.asarray(object_pos)[0, :2]))
    contact_ticks = np.flatnonzero(np.asarray(n_contacts) > 0)
    if success:
        outcome = "success"
    elif lift[-1] < -0.1:
        outcome = "knocked_off"
    elif tilt[-1] > 45.0:
        outcome = "toppled"
    elif xy_disp > 0.03 and lift[-1] < success_lift:
        outcome = "pushed"
    elif len(contact_ticks):
        outcome = "touched"
    else:
        outcome = "untouched"
    dist = np.asarray(hand_dist, dtype=float)
    return {
        "outcome": outcome,
        "first_contact_s": float((contact_ticks[0] + 1) / fps) if len(contact_ticks) else None,
        "min_hand_object_dist": float(np.nanmin(dist)) if np.isfinite(dist).any() else None,
        "final_tilt_deg": float(tilt[-1]),
        "max_tilt_deg": float(tilt.max()),
        "object_xy_displacement": xy_disp,
        "lift_then_drop": bool(success and lift[-1] < success_lift),
    }


class SimProbe:
    """Reads the object, base, hands and contacts of the live MuJoCo model (call after `connect()`).

    Hands are the `{side}_tcp` sites the sim composes into every end effector (the point that follows the tracked wrist);
    if a model has none, the first body whose name matches the side and one of palm / hand / wrist is used. The table
    is the body named `table` (all worlds so far); without it the object-table contact count is 0.
    """

    def __init__(self, mj, object_body: str | None):
        import mujoco

        self.mujoco, self.model, self.data = mujoco, mj.mj_model, mj.mj_data
        model = self.model
        body_id = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  # noqa: E731
        body_name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i))  # noqa: E731
        if object_body is not None:
            self.object_id = body_id(object_body)
            if self.object_id < 0:
                raise RuntimeError(f"No body named {object_body!r} in the sim world.")
        elif body_id("cylinder") >= 0:
            self.object_id = body_id("cylinder")
        else:
            free = sorted({int(model.jnt_bodyid[j]) for j in mj.world_object_joints})
            if len(free) != 1:
                names = [body_name(i) for i in free]
                raise RuntimeError(
                    f"Cannot pick the object: world free-joint bodies are {names}; pass --object-body."
                )
            self.object_id = free[0]
        self.object_body = body_name(self.object_id)

        robot_bodies = model.body_rootid == model.body_rootid[body_id("pelvis")]
        self.robot_geom = robot_bodies[model.geom_bodyid]
        self.object_geom = model.geom_bodyid == self.object_id
        self.table_id = body_id("table")
        self.table_geom = (
            model.geom_bodyid == self.table_id if self.table_id >= 0 else np.zeros(model.ngeom, bool)
        )

        self.hands = {}  # side -> ("site" | "body", id, name)
        for side in ("right", "left"):
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
            if site >= 0:
                self.hands[side] = ("site", site, f"{side}_tcp")
                continue
            for key in ("palm", "hand", "wrist"):
                match = [
                    i
                    for i in range(model.nbody)
                    if side in (body_name(i) or "") and key in (body_name(i) or "")
                ]
                if match:
                    self.hands[side] = ("body", match[0], body_name(match[0]))
                    break
            else:
                logger.warning("no %s hand site / body found, its position is stored as NaN", side)

    def names(self) -> dict:
        return {
            "object_body": self.object_body,
            "table_body": "table" if self.table_id >= 0 else None,
            "robot_root_body": "pelvis",
            "hands": {side: f"{kind}:{name}" for side, (kind, _, name) in self.hands.items()},
        }

    def hand_pos(self, side: str) -> np.ndarray:
        if side not in self.hands:
            return np.full(3, np.nan)
        kind, idx, _ = self.hands[side]
        return np.array(self.data.site_xpos[idx] if kind == "site" else self.data.xpos[idx])

    def read(self) -> dict:
        """One tick of the physics state."""
        data = self.data
        # The sim thread keeps stepping: take one snapshot of the contact arrays and trim them to a common length
        contact = data.contact
        g1, g2 = np.array(contact.geom1), np.array(contact.geom2)
        n = min(len(g1), len(g2))
        g1, g2 = g1[:n], g2[:n]
        obj = self.object_geom
        robot_obj = (obj[g1] & self.robot_geom[g2]) | (obj[g2] & self.robot_geom[g1])
        obj_table = (obj[g1] & self.table_geom[g2]) | (obj[g2] & self.table_geom[g1])
        object_pos = np.array(data.xpos[self.object_id])
        right, left = self.hand_pos("right"), self.hand_pos("left")
        return {
            "object_pos": object_pos,
            "object_quat": np.array(data.xquat[self.object_id]),
            "base_pos": np.array(data.qpos[:3]),
            "base_quat": np.array(data.qpos[3:7]),
            "right_hand_pos": right,
            "left_hand_pos": left,
            "hand_object_dist": float(np.linalg.norm(right - object_pos)),
            "n_contacts_robot_object": int(robot_obj.sum()),
            "n_contacts_object_table": int(obj_table.sum()),
        }


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(  # nosec B603 B607: fixed git command, no shell
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def save_trace(path: Path, ticks: list[dict], chunks: list[dict], meta: dict) -> None:
    """Stack the per-tick / per-chunk records into arrays and write them as a compressed npz (plus a `meta` JSON string)."""
    arrays = {key: np.array([tick[key] for tick in ticks]) for key in ticks[0]} if ticks else {}
    if chunks:
        arrays.update(
            chunk_tick=np.array([c["tick"] for c in chunks]),
            chunk_ready_tick=np.array([c["ready_tick"] for c in chunks]),
            chunk_infer_ms=np.array([c["infer_ms"] for c in chunks]),
            chunk_actions=np.stack([c["actions"] for c in chunks]),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    logger.info("trace saved to %s (%d ticks, %d chunks)", path, len(ticks), len(chunks))


def disable_collision_below(mj, root_body: str) -> list[str]:
    """Zero contype / conaffinity of every geom of `root_body` and its descendants; returns the bodies touched."""
    import mujoco

    model = mj.mj_model
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if root < 0:
        raise RuntimeError(f"No body named {root_body!r} in the sim model.")
    bodies = {root}
    for b in range(root + 1, model.nbody):  # MuJoCo orders bodies depth-first, children after their parent
        if model.body_parentid[b] in bodies:
            bodies.add(b)
    geoms = np.isin(model.geom_bodyid, list(bodies))
    model.geom_contype[geoms] = 0
    model.geom_conaffinity[geoms] = 0
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in sorted(bodies)]


def seed_sim_layout(robot, mj, seed: int) -> None:
    """Seed the start pose (`robot._sim_rng`) and the object jitter (`mj.world_rng`), then reset the sim so both are re-sampled.

    `_reset_sim` runs on the sim-stepping thread: `sim_env.reset()` -> `reset_world_objects()` (world_rng) and
    `_request_sim_start()` (_sim_rng). Returns once that reset has been fully applied.
    """
    done = threading.Event()
    reset_sim = robot._reset_sim

    def reset_and_signal(band) -> None:
        try:
            reset_sim(band)
        finally:
            done.set()

    robot._reset_sim = reset_and_signal
    robot._seed_sim_start(seed)
    mj.world_rng = np.random.default_rng(seed)
    robot._reset_requested = True
    if not done.wait(timeout=10.0):
        raise TimeoutError("Seeded sim reset was not processed by the sim thread.")
    robot._reset_sim = reset_sim


def run(args: argparse.Namespace, result: dict) -> None:
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    state_names, action_names, dataset_task = load_feature_names(args.policy_path, args.repo_id)
    logger.info("state dims: %d, action dims: %d", len(state_names), len(action_names))
    task = args.task or dataset_task
    if task is None:
        raise ValueError("The training dataset has no task; pass --task.")
    result["task"] = task
    logger.info("task: %r (%s)", task, "--task" if args.task else "training dataset")

    policy = GrootPolicy.from_pretrained(args.policy_path).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    image_keys = [k for k in policy.config.input_features if k.startswith(IMAGE_PREFIX)]
    logger.info("policy image inputs: %s", image_keys)

    def policy_inputs(obs: dict) -> dict:
        """This tick's state and images (copies: the worker thread reads them while the loop goes on)."""
        inputs = {OBS_STATE: np.array([obs[n] for n in state_names], dtype=np.float32)}
        for key in image_keys:
            inputs[key] = np.array(obs[key[len(IMAGE_PREFIX) :]])  # HWC uint8 RGB
        return inputs

    def predict(inputs: dict) -> np.ndarray:
        """One chunk of absolute actions, (T, action_dim). Runs on the scheduler's worker thread."""
        batch = prepare_observation_for_inference(dict(inputs), device)
        batch = {k: v for k, v in batch.items() if k == OBS_STATE or k.startswith(IMAGE_PREFIX)}
        batch["task"] = [task]
        with torch.inference_mode():
            policy.reset()
            chunk = postprocessor(policy.predict_action_chunk(preprocessor(batch)))
        if device.type == "cuda":
            torch.cuda.synchronize()
        return chunk[0].float().cpu().numpy()

    video_path = (
        Path(args.video_out)
        if args.video_out
        else (
            Path("~/Documents/g1_dev/outputs/rollouts").expanduser() / f"{datetime.now():%Y%m%d_%H%M%S}.mp4"
        )
    )
    result["video"] = str(video_path)

    robot = build_robot(args)
    writer = None
    scheduler = None
    lifts, infer_times, pelvis_z = [], [], []
    ticks, chunks = [], []  # per-tick / per-chunk trace records
    probe, trace_meta = None, {}
    obj_z0, base0, base_xy, t_success = None, None, None, None
    lift_since = None  # tick at which the current run of lifts above --success-lift started
    try:
        robot.connect()
        mj = robot.sim_env.simulator.sim_env  # holds mj_model / mj_data
        probe = SimProbe(mj, args.object_body)
        result["object_body"] = probe.object_body
        logger.info("bodies: %s", probe.names())
        if args.no_collision_left_arm:
            off = disable_collision_below(mj, "left_shoulder_pitch_link")
            logger.info("left arm collisions disabled for %d bodies: %s", len(off), off)

        if args.seed is not None:
            seed_sim_layout(robot, mj, args.seed)

        # Warm-up inference (CUDA kernels, autotuning: ~2.5 s) while the robot hangs on the band, so the first chunk of
        # the rollout does not stall the loop; re-seeded so the policy noise matches a run without it
        t0 = time.perf_counter()
        obs = {}
        while not obs:
            if time.perf_counter() - t0 > args.ready_timeout:
                raise TimeoutError("No robot observation before the timeout.")
            obs = robot.get_observation()
            time.sleep(0.05)
        predict(policy_inputs(obs))
        logger.info("warm-up inference %.0f ms", (time.perf_counter() - t0) * 1000)
        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        logger.info("waiting for band release / controller (timeout %.0fs)", args.ready_timeout)
        t0 = time.perf_counter()
        while not robot.ready_to_record():
            if time.perf_counter() - t0 > args.ready_timeout:
                raise TimeoutError("Robot not ready (elastic band still attached) before timeout.")
            time.sleep(0.1)
        logger.info("robot ready after %.1fs", time.perf_counter() - t0)

        obj_z0 = float(mj.mj_data.xpos[probe.object_id][2])
        base0 = np.array(mj.mj_data.qpos[:2])
        result["sim_start"] = list(robot.sim_start) if robot.sim_start is not None else None
        result["object_start_xyz"] = result["cylinder_start_xyz"] = [
            float(v) for v in mj.mj_data.xpos[probe.object_id]
        ]
        logger.info("sim start (x m, y m, yaw deg): %s", result["sim_start"])
        logger.info("%s start xyz: %s", probe.object_body, np.round(result["object_start_xyz"], 3).tolist())
        try:
            provenance = robot.provenance()
        except Exception as e:  # noqa: BLE001 - metadata only
            provenance = {"error": f"{type(e).__name__}: {e}"}
        trace_meta = {
            "state_names": state_names,
            "action_names": action_names,
            "args": vars(args),
            "task": task,
            "seed": args.seed,
            "sim_start": result["sim_start"],
            "object_start_xyz": result["object_start_xyz"],
            "object_start_quat": [float(v) for v in mj.mj_data.xquat[probe.object_id]],
            "bodies": probe.names(),
            "fps": args.fps,
            "execution_horizon": args.execution_horizon,
            "async_inference": not args.sync_inference,
            "success_lift": args.success_lift,
            "success_hold_s": args.success_hold_s,
            "lerobot_commit": git_commit(Path(__file__).resolve().parent),
            "provenance": provenance,
            "tick_notes": "state/action/chunk_id are the policy input / output of the tick; the physics fields are read right "
            "after send_action; t is wall time since the first tick",
        }

        writer = VideoWriter(video_path, args.fps)
        scheduler = ChunkScheduler(predict, args.execution_horizon, asynchronous=not args.sync_inference)
        base_xy = base0
        n_ticks = int(args.duration * args.fps)
        next_t = time.perf_counter()
        rollout_t0, prev_start = next_t, None
        for _ in range(n_ticks):
            loop_start = time.perf_counter()
            obs = robot.get_observation()
            if not obs:
                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_start), 0.0))
                continue

            state = np.array([obs[n] for n in state_names], dtype=np.float32)
            n_chunks = len(scheduler.chunks)
            action = scheduler.step(len(ticks), lambda obs=obs: policy_inputs(obs))
            for record in scheduler.chunks[n_chunks:]:
                infer_times.append(record.infer_s)
                chunks.append(
                    {
                        "tick": record.origin,
                        "ready_tick": record.ready_tick,
                        "infer_ms": record.infer_s * 1000,
                        "actions": record.actions,
                    }
                )
                logger.info(
                    "tick %d: chunk from tick %d, inference %.0f ms",
                    record.ready_tick,
                    record.origin,
                    record.infer_s * 1000,
                )
            robot.send_action({name: float(action[i]) for i, name in enumerate(action_names)})

            phys = probe.read()
            lifts.append(float(phys["object_pos"][2]) - obj_z0)
            base_xy = np.array(mj.mj_data.qpos[:2])
            pelvis_z.append(float(mj.mj_data.qpos[2]))
            ticks.append(
                {
                    "t": loop_start - rollout_t0,
                    "state": state,
                    "action": action.astype(np.float32),
                    "chunk_id": scheduler.chunk_id,
                    "object_tilt_deg": float(tilt_deg(phys["object_quat"])),
                    "lift": lifts[-1],
                    "loop_dt": np.nan if prev_start is None else loop_start - prev_start,
                    **phys,
                }
            )
            prev_start = loop_start
            frames = [obs[c] for c in (HEAD_CAM, TABLE_CAM) if c in obs]
            if frames:
                writer.write(np.concatenate(frames, axis=1))

            if lifts[-1] > args.success_lift:
                lift_since = len(lifts) - 1 if lift_since is None else lift_since
            else:
                lift_since = None
            if (
                t_success is None
                and lift_since is not None
                and len(lifts) - lift_since >= args.success_hold_s * args.fps
            ):
                t_success = len(lifts) / args.fps
                logger.info(
                    "success: lift %.3f m held %.1fs, at %.1fs", lifts[-1], args.success_hold_s, t_success
                )
            if (
                args.stop_after_success_s is not None
                and t_success is not None
                and len(lifts) / args.fps - t_success >= args.stop_after_success_s
            ):
                logger.info("stopping %.1fs after success", args.stop_after_success_s)
                break

            next_t += 1.0 / args.fps
            precise_sleep(max(next_t - time.perf_counter(), 0.0))
            if next_t < time.perf_counter() - 0.5:  # fell far behind (e.g. first inference): resync
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        # Filled in `finally` so a run that raises mid-rollout still reports what it measured, and saved before the
        # teardown: the ZMQ camera shutdown can abort the process (libzmq assertion), which no `except` catches
        if lifts:
            result.update(
                success=t_success is not None,
                max_lift=max(lifts),
                final_lift=lifts[-1],
                time_to_success_s=t_success,
                ticks=len(lifts),
                duration_s=len(lifts) / args.fps,
                base_travel_m=float(np.linalg.norm(base_xy - base0)),
                min_pelvis_z=min(pelvis_z),
                fell=bool(min(pelvis_z) < 0.5),
            )
            column = lambda key: np.array([tick[key] for tick in ticks])  # noqa: E731
            result.update(
                classify_outcome(
                    t_success is not None,
                    column("lift"),
                    column("object_pos"),
                    column("object_tilt_deg"),
                    column("n_contacts_robot_object"),
                    column("hand_object_dist"),
                    args.fps,
                    args.success_lift,
                )
            )
        if scheduler is not None:
            scheduler.close()
            result["async_inference"] = scheduler.asynchronous
            result["blocked_ticks"] = scheduler.blocked_ticks
            result["stale_actions_skipped"] = int(sum(c.ready_tick - c.origin for c in scheduler.chunks))
        if infer_times:
            result["inference_mean_ms"] = float(np.mean(infer_times) * 1000)
            result["inference_max_ms"] = float(np.max(infer_times) * 1000)

        save_result(args, result)
        if args.trace_npz and ticks:
            try:
                save_trace(Path(args.trace_npz), ticks, chunks, {**trace_meta, "result": result})
            except Exception:  # noqa: BLE001 - the trace must not cost the video / teardown
                logger.exception("could not save the trace")
        if writer is not None:
            writer.close()
            logger.info("video saved to %s", video_path)
        # The ZMQ camera shutdown can also hang (a read thread that never stops); everything is saved by now, so give it
        # DISCONNECT_TIMEOUT_S and let main() hard-exit instead of waiting for the eval script's timeout
        teardown = threading.Thread(target=robot.disconnect, daemon=True)
        teardown.start()
        teardown.join(DISCONNECT_TIMEOUT_S)
        if teardown.is_alive():
            logger.warning(
                "robot.disconnect() still running after %.0f s, giving up on it", DISCONNECT_TIMEOUT_S
            )
            result["teardown_hung"] = True
    if lifts:
        print(f"ticks run: {len(lifts)} ({len(lifts) / args.fps:.1f}s)")
        if infer_times:
            print(
                f"inference per chunk: mean {np.mean(infer_times) * 1000:.0f} ms, max {np.max(infer_times) * 1000:.0f} ms"
            )
        print(f"{probe.object_body} start z: {obj_z0:.3f} m")
        print(f"max lift: {max(lifts):.3f} m, final lift: {lifts[-1]:.3f} m")
        print(f"success (> {args.success_lift} m held {args.success_hold_s}s): {t_success is not None}")
        print(f"outcome: {result['outcome']}")
        print(f"base distance travelled (xy, start to end): {np.linalg.norm(base_xy - base0):.3f} m")


DISCONNECT_TIMEOUT_S = 15.0


def save_result(args: argparse.Namespace, result: dict) -> None:
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        logger.info("result saved to %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    result = {
        "policy_path": args.policy_path,
        "seed": args.seed,
        "success": False,
        "max_lift": None,
        "final_lift": None,
        "time_to_success_s": None,
        "ticks": 0,
        "duration_s": 0.0,
        "sim_start": None,
        "cylinder_start_xyz": None,  # = object_start_xyz, kept for old result readers
        "object_body": None,
        "object_start_xyz": None,
        "task": None,
        "outcome": None,
        "first_contact_s": None,
        "min_hand_object_dist": None,
        "final_tilt_deg": None,
        "max_tilt_deg": None,
        "object_xy_displacement": None,
        "lift_then_drop": None,
        "base_travel_m": None,
        "min_pelvis_z": None,
        "fell": None,
        "inference_mean_ms": None,
        "inference_max_ms": None,
        "video": None,
        "error": None,
    }
    try:
        run(args, result)
    except Exception as e:  # noqa: BLE001 - recorded in the result, then exit nonzero
        logger.exception("rollout failed")
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        save_result(args, result)
    if result.get("teardown_hung"):  # the stuck camera thread would also block interpreter exit
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if result["error"] else 0)
    if result["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
