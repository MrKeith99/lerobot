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
(23dof, AmazingHand, pan/tilt D455, GR00T locomotion controller, ``pick_cylinder`` world with the
``head_camera`` and ``table_view`` ZMQ cameras), waits until the elastic band is released, then every
``--execution-horizon`` ticks runs preprocessor -> ``predict_action_chunk`` -> postprocessor on the current
observation and sends the resulting absolute actions one per tick at ``--fps``. The cylinder height (read
from the MuJoCo state) is tracked to report the maximum / final lift. A run succeeds when the cylinder stays more than
``--success-lift`` above its start for ``--success-hold-s`` in a row (lifted and held, so a knock or bounce does not count).
An mp4 with both cameras side by side is saved.

Run it headless (no display) with the same environment as the XR sim recording:

```shell
export CYCLONEDDS_HOME=~/Documents/g1_dev/cyclonedds/install LD_LIBRARY_PATH=~/miniconda3/envs/lerobot/lib \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl UNITREE_G1_MUJOCO_ONSCREEN=0 SDL_VIDEODRIVER=dummy
unset DISPLAY
python examples/unitree_g1/groot_sim_rollout.py \
    --policy-path outputs/train/my_run/checkpoints/last/pretrained_model \
    --world pick_cylinder --duration 40 --execution-horizon 16 \
    --seed 3 --stop-after-success-s 3 --result-json outputs/eval/seed_03.json
```

The defaults match the XR sim recording (`guides/run_xr_sim_record.sh`): world jitter 0.08, robot started 1.3 m behind the
spawn (``--start-offset -1.3 0 0``) with a (0.15 m, 15 deg) random start, cylinder re-placed 3 s after the band release.
``--seed`` makes the policy noise, the start pose and the cylinder jitter reproducible (same layout for every checkpoint);
``--result-json`` writes one machine-readable result per run (see `guides/run_groot_eval.sh`).
"""

import argparse
import json
import logging
import sys
import threading
import time
from collections import deque
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
    parser.add_argument("--task", default="Pick up the cylinder from the table with the right hand")
    parser.add_argument("--world", default="pick_cylinder", help="Sim world name.")
    parser.add_argument("--randomize", type=float, default=0.08, help="World object xy jitter in metres.")
    parser.add_argument(
        "--start-offset",
        type=float,
        nargs=3,
        default=[-1.3, 0.0, 0.0],
        metavar=("X", "Y", "YAW"),
        help="Robot start relative to the spawn (m, m, deg); default: 1.3 m back so it has to walk up to the table.",
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
        "--seed", type=int, default=None, help="Seed torch / numpy and the sim start + cylinder jitter."
    )
    parser.add_argument("--duration", type=float, default=40.0, help="Rollout length in seconds.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--execution-horizon", type=int, default=16, help="Actions executed per predicted chunk."
    )
    parser.add_argument(
        "--band-release-s", type=float, default=2.0, help="Seconds after connect to release the band."
    )
    parser.add_argument(
        "--success-lift", type=float, default=0.05, help="Cylinder lift (m) counting as success."
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
    return parser.parse_args()


def load_feature_names(policy_path: str, repo_id: str | None) -> tuple[list[str], list[str]]:
    """State and action names in training-dataset order."""
    from lerobot.datasets import LeRobotDatasetMetadata

    if repo_id is None:
        repo_id = json.loads((Path(policy_path) / "train_config.json").read_text())["dataset"]["repo_id"]
    features = LeRobotDatasetMetadata(repo_id).features
    return list(features[OBS_STATE]["names"]), list(features[ACTION]["names"])


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


def seed_sim_layout(robot, mj, seed: int) -> None:
    """Seed the start pose (`robot._sim_rng`) and the cylinder jitter (`mj.world_rng`), then reset the sim so both are re-sampled.

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

    state_names, action_names = load_feature_names(args.policy_path, args.repo_id)
    logger.info("state dims: %d, action dims: %d", len(state_names), len(action_names))

    policy = GrootPolicy.from_pretrained(args.policy_path).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    image_keys = [k for k in policy.config.input_features if k.startswith(IMAGE_PREFIX)]
    logger.info("policy image inputs: %s", image_keys)

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
    lifts, infer_times, pelvis_z = [], [], []
    cyl_z0, base0, base_xy, t_success = None, None, None, None
    lift_since = None  # tick at which the current run of lifts above --success-lift started
    try:
        robot.connect()
        mj = robot.sim_env.simulator.sim_env  # holds mj_model / mj_data
        import mujoco

        cyl_id = mujoco.mj_name2id(mj.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
        if cyl_id < 0:
            raise RuntimeError("No body named 'cylinder' in the sim world.")

        if args.seed is not None:
            seed_sim_layout(robot, mj, args.seed)

        logger.info("waiting for band release / controller (timeout %.0fs)", args.ready_timeout)
        t0 = time.perf_counter()
        while not robot.ready_to_record():
            if time.perf_counter() - t0 > args.ready_timeout:
                raise TimeoutError("Robot not ready (elastic band still attached) before timeout.")
            time.sleep(0.1)
        logger.info("robot ready after %.1fs", time.perf_counter() - t0)

        cyl_z0 = float(mj.mj_data.xpos[cyl_id][2])
        base0 = np.array(mj.mj_data.qpos[:2])
        result["sim_start"] = list(robot.sim_start) if robot.sim_start is not None else None
        result["cylinder_start_xyz"] = [float(v) for v in mj.mj_data.xpos[cyl_id]]
        logger.info("sim start (x m, y m, yaw deg): %s", result["sim_start"])
        logger.info("cylinder start xyz: %s", np.round(result["cylinder_start_xyz"], 3).tolist())

        writer = VideoWriter(video_path, args.fps)
        queue: deque[np.ndarray] = deque()
        base_xy = base0
        n_ticks = int(args.duration * args.fps)
        next_t = time.perf_counter()
        for tick in range(n_ticks):
            loop_start = time.perf_counter()
            obs = robot.get_observation()
            if not obs:
                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_start), 0.0))
                continue

            if len(queue) == 0:
                inputs = {OBS_STATE: np.array([obs[n] for n in state_names], dtype=np.float32)}
                for key in image_keys:
                    inputs[key] = np.ascontiguousarray(obs[key[len(IMAGE_PREFIX) :]])  # HWC uint8 RGB
                batch = prepare_observation_for_inference(inputs, device)
                batch = {k: v for k, v in batch.items() if k == OBS_STATE or k.startswith(IMAGE_PREFIX)}
                batch["task"] = [args.task]
                t_inf = time.perf_counter()
                with torch.inference_mode():
                    policy.reset()
                    chunk = policy.predict_action_chunk(preprocessor(batch))
                    chunk = postprocessor(chunk)  # absolute actions, (1, T, action_dim)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t_inf
                infer_times.append(dt)
                queue.extend(chunk[0, : args.execution_horizon].float().cpu().numpy())
                logger.info("tick %d: chunk inference %.0f ms", tick, dt * 1000)

            action = queue.popleft()
            robot.send_action({name: float(action[i]) for i, name in enumerate(action_names)})

            lifts.append(float(mj.mj_data.xpos[cyl_id][2]) - cyl_z0)
            base_xy = np.array(mj.mj_data.qpos[:2])
            pelvis_z.append(float(mj.mj_data.qpos[2]))
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
        if infer_times:
            result["inference_mean_ms"] = float(np.mean(infer_times) * 1000)
            result["inference_max_ms"] = float(np.max(infer_times) * 1000)

        save_result(args, result)
        if writer is not None:
            writer.close()
            logger.info("video saved to %s", video_path)
        robot.disconnect()
    if lifts:
        print(f"ticks run: {len(lifts)} ({len(lifts) / args.fps:.1f}s)")
        if infer_times:
            print(
                f"inference per chunk: mean {np.mean(infer_times) * 1000:.0f} ms, max {np.max(infer_times) * 1000:.0f} ms"
            )
        print(f"cylinder start z: {cyl_z0:.3f} m")
        print(f"max lift: {max(lifts):.3f} m, final lift: {lifts[-1]:.3f} m")
        print(f"success (> {args.success_lift} m held {args.success_hold_s}s): {t_success is not None}")
        print(f"base distance travelled (xy, start to end): {np.linalg.norm(base_xy - base0):.3f} m")


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
        "cylinder_start_xyz": None,
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
    if result["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
