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
from the MuJoCo state) is tracked to report the maximum / final lift and whether it exceeds ``--success-lift``.
An mp4 with both cameras side by side is saved.

Run it headless (no display) with the same environment as the XR sim recording:

```shell
export CYCLONEDDS_HOME=~/Documents/g1_dev/cyclonedds/install LD_LIBRARY_PATH=~/miniconda3/envs/lerobot/lib \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl UNITREE_G1_MUJOCO_ONSCREEN=0 SDL_VIDEODRIVER=dummy
unset DISPLAY
python examples/unitree_g1/groot_sim_rollout.py \
    --policy-path outputs/train/my_run/checkpoints/last/pretrained_model \
    --world pick_cylinder --duration 60 --execution-horizon 16
```
"""

import argparse
import json
import logging
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True, help="Path to a checkpoint's `pretrained_model` dir.")
    parser.add_argument("--task", default="Pick up the cylinder from the table with the right hand")
    parser.add_argument("--world", default="pick_cylinder", help="Sim world name.")
    parser.add_argument("--randomize", type=float, default=0.02, help="World object xy jitter in metres.")
    parser.add_argument("--duration", type=float, default=60.0, help="Rollout length in seconds.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--execution-horizon", type=int, default=16, help="Actions executed per predicted chunk.")
    parser.add_argument("--band-release-s", type=float, default=2.0, help="Seconds after connect to release the band.")
    parser.add_argument("--success-lift", type=float, default=0.05, help="Cylinder lift (m) counting as success.")
    parser.add_argument("--video-out", default=None, help="mp4 path (default outputs/rollouts/<timestamp>.mp4).")
    parser.add_argument("--no-gravity-compensation", action="store_true", help="Disable the arm gravity feed-forward.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ready-timeout", type=float, default=60.0, help="Max seconds to wait for the band release.")
    parser.add_argument("--repo-id", default=None, help="Training dataset (for state/action names); default: from train_config.json.")
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    device = torch.device(args.device)

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

    video_path = Path(args.video_out) if args.video_out else (
        Path("~/Documents/g1_dev/outputs/rollouts").expanduser() / f"{datetime.now():%Y%m%d_%H%M%S}.mp4"
    )

    robot = build_robot(args)
    writer = None
    try:
        robot.connect()
        mj = robot.sim_env.simulator.sim_env  # holds mj_model / mj_data
        import mujoco

        cyl_id = mujoco.mj_name2id(mj.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
        if cyl_id < 0:
            raise RuntimeError("No body named 'cylinder' in the sim world.")

        logger.info("waiting for band release / controller (timeout %.0fs)", args.ready_timeout)
        t0 = time.perf_counter()
        while not robot.ready_to_record():
            if time.perf_counter() - t0 > args.ready_timeout:
                raise TimeoutError("Robot not ready (elastic band still attached) before timeout.")
            time.sleep(0.1)
        logger.info("robot ready after %.1fs", time.perf_counter() - t0)

        cyl_z0 = float(mj.mj_data.xpos[cyl_id][2])
        base0 = np.array(mj.mj_data.qpos[:2])
        logger.info("cylinder start z=%.3f", cyl_z0)

        writer = VideoWriter(video_path, args.fps)
        queue: deque[np.ndarray] = deque()
        lifts, infer_times = [], []
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
            frames = [obs[c] for c in (HEAD_CAM, TABLE_CAM) if c in obs]
            if frames:
                writer.write(np.concatenate(frames, axis=1))

            next_t += 1.0 / args.fps
            precise_sleep(max(next_t - time.perf_counter(), 0.0))
            if next_t < time.perf_counter() - 0.5:  # fell far behind (e.g. first inference): resync
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        logger.info("interrupted")
        lifts = locals().get("lifts", [])
    finally:
        if writer is not None:
            writer.close()
            logger.info("video saved to %s", video_path)
        robot.disconnect()

    if lifts:
        max_lift, final_lift = max(lifts), lifts[-1]
        print(f"ticks run: {len(lifts)} ({len(lifts) / args.fps:.1f}s)")
        if infer_times:
            print(f"inference per chunk: mean {np.mean(infer_times) * 1000:.0f} ms, max {np.max(infer_times) * 1000:.0f} ms")
        print(f"cylinder start z: {cyl_z0:.3f} m")
        print(f"max lift: {max_lift:.3f} m, final lift: {final_lift:.3f} m")
        print(f"success (> {args.success_lift} m): {max_lift > args.success_lift}")
        print(f"base distance travelled (xy, start to end): {np.linalg.norm(base_xy - base0):.3f} m")


if __name__ == "__main__":
    main()
