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

"""Deferred episode deletion for datasets that are still being recorded.

A saved v3.0 episode cannot be removed in place (its frames share parquet / video files with the others),
so `mark_episode_deleted` only appends its index to `<root>/meta/deleted_episodes.json` (a tombstone, crash
safe) and `apply_deleted_episodes` rewrites the dataset without the marked episodes once it is finalized.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from lerobot.utils.provenance import PROVENANCE_FILE

logger = logging.getLogger(__name__)

DELETED_EPISODES_FILE = "meta/deleted_episodes.json"
_TMP_SUFFIX = ".tmp-delete"
_ASIDE_SUFFIX = ".bak-delete"


def read_deleted_episodes(root: str | Path) -> list[int]:
    """Episode indices marked as deleted under `root` (sorted, empty without a tombstone file)."""
    path = Path(root) / DELETED_EPISODES_FILE
    if not path.is_file():
        return []
    return sorted({int(i) for i in json.loads(path.read_text())["episodes"]})


def mark_episode_deleted(root: str | Path, episode_index: int) -> list[int]:
    """Add `episode_index` to the tombstone file, atomically; returns all marked indices."""
    path = Path(root) / DELETED_EPISODES_FILE
    episodes = sorted({*read_deleted_episodes(root), int(episode_index)})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"episodes": episodes}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return episodes


def apply_deleted_episodes(root: str | Path, repo_id: str | None = None) -> bool:
    """Rewrite the finalized dataset at `root` without its marked episodes.

    The new dataset is built next to `root`, then swapped in (original renamed aside, new one renamed into
    place, aside copy removed). On any failure the original and its tombstone file stay as they were and the
    reason is logged. Returns True when `root` holds no pending deletion afterwards.
    """
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root).resolve()
    tmp, aside = (root.with_name(root.name + suffix) for suffix in (_TMP_SUFFIX, _ASIDE_SUFFIX))
    reapply = f"python examples/unitree_g1/apply_deleted_episodes.py --root {root}"

    if not root.exists() and aside.exists():  # crashed between the two renames
        aside.rename(root)
    marked = read_deleted_episodes(root)
    if not marked:
        return True
    if aside.exists():
        logger.error(f"{aside} is left over from an earlier run; check it, remove it and re-apply: {reapply}")
        return False
    repo_id = repo_id or f"local/{root.name}"

    try:
        dataset = LeRobotDataset(repo_id, root=root)
        total = dataset.meta.total_episodes
        deleted = [i for i in marked if i < total]
        if len(deleted) != len(marked):
            logger.warning(
                f"Ignoring deleted episodes {sorted(set(marked) - set(deleted))}: dataset has {total}"
            )
        if len(deleted) == total:
            logger.warning(
                f"Every episode of {root} is marked as deleted; keeping the dataset as is with "
                f"{DELETED_EPISODES_FILE} (an empty dataset cannot be written)."
            )
            return False
        if tmp.exists():
            shutil.rmtree(tmp)
        cleaned = delete_episodes(dataset, deleted, output_dir=tmp, repo_id=repo_id)
        if cleaned.meta.total_episodes != total - len(deleted):
            raise RuntimeError(f"expected {total - len(deleted)} episodes, got {cleaned.meta.total_episodes}")
        del cleaned
        provenance = root / PROVENANCE_FILE
        if provenance.is_file() and not (tmp / PROVENANCE_FILE).exists():
            shutil.copy2(provenance, tmp / PROVENANCE_FILE)
        (tmp / DELETED_EPISODES_FILE).unlink(missing_ok=True)

        root.rename(aside)
        try:
            tmp.rename(root)
        except Exception:
            aside.rename(root)
            raise
    except Exception:
        logger.exception(
            f"Could not remove episodes {marked} from {root}; the dataset and {DELETED_EPISODES_FILE} are "
            f"left intact. Re-apply with: {reapply}"
        )
        shutil.rmtree(tmp, ignore_errors=True)
        return False

    shutil.rmtree(aside, ignore_errors=True)
    logger.info(f"Removed episodes {deleted} from {root} ({total - len(deleted)} left)")
    return True
