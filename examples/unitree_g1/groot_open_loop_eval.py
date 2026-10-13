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

"""Open-loop evaluation of a LeRobot GR00T N1.7 checkpoint on a LeRobot v3.0 dataset.

Mirrors Isaac-GR00T's ``gr00t/eval/open_loop_eval.py``. For each requested episode the script
steps through the frames every ``--execution-horizon`` frames, feeds the recorded observation
(state, images, task) to the policy through the saved pre/post-processors, keeps the first
``execution_horizon`` predicted (absolute) actions and compares them with the ground-truth
``action`` of the dataset. It reports MSE / MAE per episode and overall, plus the MAE of every
action dimension, and can optionally save a GT-vs-prediction plot per episode.

Example:

```shell
python examples/unitree_g1/groot_open_loop_eval.py \
    --policy-path outputs/train/my_run/checkpoints/last/pretrained_model \
    --repo-id k-valentin/g1_xr_sim_pick_cylinder_20261013_110335_ego \
    --episodes 0 1 --execution-horizon 16 --plot-dir outputs/open_loop_plots
```
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True, help="Path to a checkpoint's `pretrained_model` dir.")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id.")
    parser.add_argument("--root", default=None, help="Optional local dataset root.")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0], help="Episode indices to evaluate.")
    parser.add_argument("--execution-horizon", type=int, default=16, help="Frames per predicted chunk used.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional frame cap per episode.")
    parser.add_argument("--plot-dir", default=None, help="If set, save one PNG per episode here.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_batch(item: dict, device: torch.device) -> dict:
    """Turn one dataset item into a batch of size 1, as the preprocessor expects.

    Dataset items already hold float32 state and float CHW images in [0, 1], so we only add the
    batch dimension and keep ``task`` as a string (the language step broadcasts it to the batch).
    """
    batch = {}
    for key, value in item.items():
        if key == OBS_STATE or key.startswith("observation.images."):
            batch[key] = value.unsqueeze(0).to(device)
    batch["task"] = [item["task"]]
    return batch


def plot_episode(gt: np.ndarray, pred: np.ndarray, names: list[str], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_dims = gt.shape[1]
    cols = 5
    rows = int(np.ceil(n_dims / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.2 * rows), squeeze=False)
    for d, ax in enumerate(axes.flat):
        if d >= n_dims:
            ax.axis("off")
            continue
        ax.plot(gt[:, d], label="GT", linewidth=1)
        ax.plot(pred[:, d], label="pred", linewidth=1)
        ax.set_title(names[d], fontsize=8)
        ax.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    policy = GrootPolicy.from_pretrained(args.policy_path).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=args.episodes, video_backend="pyav")
    action_names = dataset.meta.features[ACTION]["names"]
    # `dataset` only holds the requested episodes, so locate each one's frames via its index column.
    episode_col = np.asarray(dataset.hf_dataset["episode_index"])
    # Ground-truth actions straight from the parquet columns (indexing `dataset` would decode video).
    gt_actions = np.stack([np.asarray(a, dtype=np.float32) for a in dataset.hf_dataset[ACTION]])

    all_errors = []
    for ep in args.episodes:
        frame_ids = np.flatnonzero(episode_col == ep)
        if args.max_steps is not None:
            frame_ids = frame_ids[: args.max_steps]
        if len(frame_ids) == 0:
            raise ValueError(f"Episode {ep} not found in dataset.")

        gt_chunks, pred_chunks = [], []
        for start in range(0, len(frame_ids), args.execution_horizon):
            ids = frame_ids[start : start + args.execution_horizon]
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            batch = build_batch(dataset[int(ids[0])], device)
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(preprocessor(batch))
                chunk = postprocessor(chunk)  # absolute actions, (1, T, action_dim)
            pred_chunks.append(chunk[0, : len(ids)].float().cpu().numpy())
            gt_chunks.append(gt_actions[ids])

        gt, pred = np.concatenate(gt_chunks), np.concatenate(pred_chunks)
        err = pred - gt
        all_errors.append(err)
        print(
            f"episode {ep}: frames={len(gt)} MSE={np.mean(err**2):.6f} MAE={np.mean(np.abs(err)):.6f}"
        )
        if args.plot_dir:
            plot_episode(
                gt, pred, action_names, f"episode {ep} (horizon {args.execution_horizon})",
                Path(args.plot_dir) / f"episode_{ep:06d}.png",
            )

    errors = np.concatenate(all_errors)
    print(f"overall: frames={len(errors)} MSE={np.mean(errors**2):.6f} MAE={np.mean(np.abs(errors)):.6f}")
    print("per-dimension MAE:")
    for name, mae in zip(action_names, np.mean(np.abs(errors), axis=0), strict=True):
        print(f"  {name}: {mae:.6f}")


if __name__ == "__main__":
    main()
