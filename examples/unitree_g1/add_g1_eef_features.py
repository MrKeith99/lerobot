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

"""Append end-effector poses (forward kinematics) to a recorded Unitree G1 LeRobot v3.0 dataset.

Writes a NEW local dataset (the input is never modified) whose ``observation.state`` and ``action``
get 18 extra trailing columns, in this order, with the same names in both vectors::

    kLeftEef.x, kLeftEef.y, kLeftEef.z, kLeftEef.r0 ... kLeftEef.r5,
    kRightEef.x, kRightEef.y, kRightEef.z, kRightEef.r0 ... kRightEef.r5

* Frame: pelvis (what ``xr_teleoperate.core.G1ArmIK.forward`` returns).
* Point: the wrist flange, i.e. the ``{side}_tcp`` pose moved back along the tcp's own x axis by the
  tcp offset (0.05 m for the AmazingHand), which is the end-effector mount origin.
* Rotation: ``rot6d`` = the first two ROWS of the rotation matrix, flattened (``r0..r2 = R[0]``,
  ``r3..r5 = R[1]``), the GR00T convention (``homogeneous_to_xyz_rot6d``).
* ``observation.state`` EEF = FK of the measured arm joints, ``action`` EEF = FK of the commanded
  arm joints. Joints are matched by name; for the 23dof body only the 5 real joints per side enter
  FK (the wrist pitch / yaw slots are phantom).

Videos, episodes, tasks and every other column are copied unchanged (no re-encode). ``info.json``
shapes / names, ``stats.json`` and the per-episode stats of the two changed features are recomputed
with LeRobot's own stats code, and ``meta/provenance.json`` gets an ``eef_features`` entry. The body
and end effector come from the input's ``meta/provenance.json`` (``embodiment``) when present, else
from ``--body`` / ``--end-effector``.

Example:

```shell
python examples/unitree_g1/add_g1_eef_features.py \
    --repo-id k-valentin/g1_xr_sim_pick_cylinder_20261013_110335
# -> k-valentin/g1_xr_sim_pick_cylinder_20261013_110335_eef (under $HF_LEROBOT_HOME)
```

Drop the table camera afterwards with ``lerobot-edit-dataset`` (``remove_feature``) if the policy
only uses the head camera.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import _write_parquet
from lerobot.datasets.io_utils import write_stats
from lerobot.policies.groot.utils import homogeneous_to_xyz_rot6d
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.utils.provenance import PROVENANCE_FILE, code_info, lerobot_info

SIDES = ("Left", "Right")
EEF_SUFFIXES = ("x", "y", "z") + tuple(f"r{i}" for i in range(6))
EEF_NAMES = [f"k{side}Eef.{suffix}" for side in SIDES for suffix in EEF_SUFFIXES]
# The AmazingHand `{side}_tcp` site is 0.05 m along the mount's x axis (identity rotation). Used only when
# the FK model has no `{side}_ee_mount` frame to read the offset from.
DEFAULT_TCP_OFFSET = 0.05
SCRIPT = "examples/unitree_g1/add_g1_eef_features.py"


def arm_key(joint_name: str) -> str:
    """`left_shoulder_pitch_joint` -> `kLeftShoulderPitch.q`."""
    words = joint_name.removesuffix("_joint").split("_")
    return "k" + "".join(word.capitalize() for word in words) + ".q"


def flange_pose(tcp_pose: np.ndarray, tcp_offset: float = DEFAULT_TCP_OFFSET) -> np.ndarray:
    """Wrist flange pose: the tcp pose moved back `tcp_offset` m along the tcp's own x axis."""
    back = np.eye(4)
    back[0, 3] = -tcp_offset
    return tcp_pose @ back


def eef_features(
    forward, joint_indices: np.ndarray, q: np.ndarray, tcp_offsets: dict[str, float]
) -> np.ndarray:
    """(N, 18) flange xyz + rot6d per side from the (N, D) joint vectors `q`.

    `forward(q_arm)` returns ``{"left": 4x4, "right": 4x4}`` tcp poses in the pelvis frame and
    `joint_indices` selects the FK joints (in FK order) from the columns of `q`.
    """
    out = np.empty((len(q), 9 * len(SIDES)), dtype=np.float64)
    for i, row in enumerate(q):
        poses = forward(row[joint_indices].astype(np.float64))
        out[i] = np.concatenate(
            [
                homogeneous_to_xyz_rot6d(flange_pose(poses[side.lower()], tcp_offsets[side.lower()]))
                for side in SIDES
            ]
        )
    return out.astype(np.float32)


def fk_tcp_offsets(arm_ik) -> dict[str, float]:
    """The tcp offset along the mount's x axis, read off the FK model (mount frame -> tcp frame).

    Falls back to `DEFAULT_TCP_OFFSET` when the model has no `{side}_ee_mount` frame (bare wrist).
    """
    import pinocchio as pin

    model, data = arm_ik.model, arm_ik.data
    pin.framesForwardKinematics(model, data, np.zeros(model.nq))
    offsets = {}
    for side, tcp_id in arm_ik.frame_ids.items():
        mount_id = model.getFrameId(f"{side}_ee_mount")
        if mount_id >= len(model.frames):
            offsets[side] = DEFAULT_TCP_OFFSET
            continue
        rel = data.oMf[mount_id].inverse() * data.oMf[tcp_id]
        if not np.allclose(rel.rotation, np.eye(3), atol=1e-6) or not np.allclose(
            rel.translation[1:], 0.0, atol=1e-6
        ):
            raise RuntimeError(f"{side}_tcp is not a pure x offset from {side}_ee_mount: {rel}")
        offsets[side] = float(rel.translation[0])
    return offsets


def resolve_sim_root(args: argparse.Namespace, provenance: dict) -> tuple[Path | None, dict]:
    """The sim repo G1ArmIK builds its model from: --sim-root, else the recorded sim, else the default."""
    if args.sim_root:
        return Path(args.sim_root), {"path": str(args.sim_root)}
    sim = provenance.get("sim") or {}
    if sim.get("repo_id") and sim.get("revision"):
        from xr_teleoperate.core.embodiment import SIM_REPO_ID, sim_root

        if sim["repo_id"] == SIM_REPO_ID:
            root = sim_root(revision=sim["revision"])
            return root, code_info(root)
    elif sim.get("path") and Path(sim["path"]).is_dir():
        return Path(sim["path"]), sim
    from xr_teleoperate.core.embodiment import sim_root

    root = sim_root()
    return root, code_info(root)


def episode_stats_by_index(df: pd.DataFrame, features: dict) -> dict[int, dict]:
    """Per-episode stats of the two changed features, via LeRobot's stats code."""
    out = {}
    for ep_idx, ep_df in df.groupby("episode_index"):
        data = {key: np.stack(ep_df[key].to_numpy()) for key in (OBS_STATE, ACTION)}
        out[int(ep_idx)] = compute_episode_stats(data, {k: features[k] for k in data})
    return out


def rewrite_episode_stats(root: Path, ep_stats: dict[int, dict]) -> None:
    """Replace the `stats/{observation.state,action}/*` columns of `meta/episodes` in place."""
    for path in sorted((root / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        episodes = table["episode_index"].to_pylist()
        for key in (OBS_STATE, ACTION):
            for stat in ep_stats[episodes[0]][key]:
                col = f"stats/{key}/{stat}"
                idx = table.schema.get_field_index(col)
                values = [np.asarray(ep_stats[ep][key][stat]).tolist() for ep in episodes]
                field = table.schema.field(idx)
                table = table.set_column(idx, field, pa.array(values, field.type))
        pq.write_table(table, path, compression="snappy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="Input LeRobot dataset repo id.")
    parser.add_argument("--root", default=None, help="Optional local root of the input dataset.")
    parser.add_argument("--output-repo-id", default=None, help="Output repo id (default: <repo-id>_eef).")
    parser.add_argument("--output-root", default=None, help="Output root (default: $HF_LEROBOT_HOME/<id>).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dataset.")
    parser.add_argument("--body", default="23dof", choices=["23dof", "29dof"], help="If not in provenance.")
    parser.add_argument("--end-effector", default="amazing_hand", help="If not in provenance.")
    parser.add_argument("--sim-root", default=None, help="Sim repo checkout (default: recorded / Hub).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    src_root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    out_repo_id = args.output_repo_id or f"{args.repo_id}_eef"
    out_root = Path(args.output_root) if args.output_root else HF_LEROBOT_HOME / out_repo_id
    if out_root.resolve() == src_root.resolve():
        raise ValueError("The output must differ from the input dataset.")
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to replace it.")
        shutil.rmtree(out_root)

    info = json.loads((src_root / "meta/info.json").read_text())
    features = info["features"]
    names = {key: list(features[key]["names"]) for key in (OBS_STATE, ACTION)}
    if any(n in names[key] for key in names for n in EEF_NAMES):
        raise ValueError("The input already has end-effector features.")

    prov_path = src_root / PROVENANCE_FILE
    provenance = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    embodiment = provenance.get("embodiment") or {}
    body = embodiment.get("body", args.body)
    end_effector = embodiment.get("end_effector", args.end_effector)
    sim_root_path, sim_info = resolve_sim_root(args, provenance)

    from xr_teleoperate.core import G1ArmIK

    arm_ik = G1ArmIK(body, end_effector, sim_root=sim_root_path)
    tcp_offsets = fk_tcp_offsets(arm_ik)
    logging.info(f"FK: {body} / {end_effector}, sim {sim_info}, tcp offsets {tcp_offsets}")
    joint_idx = {
        key: np.array([names[key].index(arm_key(j)) for j in arm_ik.joint_names]) for key in names
    }

    # Copy everything (videos, episodes, tasks, ...) and then rewrite only what changes.
    shutil.copytree(src_root, out_root)
    for key in names:
        features[key]["names"] = names[key] + EEF_NAMES
        features[key]["shape"] = [len(names[key]) + len(EEF_NAMES)]
    (out_root / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")
    meta = LeRobotDatasetMetadata(out_repo_id, root=out_root)

    frames = []
    for path in sorted((out_root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(path).reset_index(drop=True)
        for key in names:
            q = np.stack(df[key].to_numpy())
            eef = eef_features(arm_ik.forward, joint_idx[key], q, tcp_offsets)
            df[key] = list(np.concatenate([q, eef], axis=1).astype(np.float32))
        _write_parquet(df, path, meta)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    ep_stats = episode_stats_by_index(df, features)
    rewrite_episode_stats(out_root, ep_stats)
    stats = json.loads((out_root / "meta/stats.json").read_text())
    new = aggregate_stats(list(ep_stats.values()))
    stats.update({key: new[key] for key in names})
    write_stats(stats, out_root)

    provenance["eef_features"] = {
        "script": SCRIPT,
        "lerobot_commit": lerobot_info().get("commit"),
        "frame": "pelvis",
        "point": f"wrist flange (tcp - {tcp_offsets['left']:g} m x)",
        "rot6d": "rows",
        "body": body,
        "end_effector": end_effector,
        "sim": sim_info,
        "source": args.repo_id,
    }
    (out_root / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {out_repo_id} ({len(df)} frames) to {out_root}")


if __name__ == "__main__":
    main()
