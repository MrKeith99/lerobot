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
"""Tests for the deferred episode deletion (tombstone file) used by lerobot-record."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import deleted_episodes  # noqa: E402
from lerobot.datasets.deleted_episodes import (  # noqa: E402
    DELETED_EPISODES_FILE,
    apply_deleted_episodes,
    mark_episode_deleted,
    read_deleted_episodes,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

REPO_ID = "local/deleted_episodes_test"


@pytest.fixture
def root(tmp_path, empty_lerobot_dataset_factory):
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    }
    root = tmp_path / "dataset"
    dataset = empty_lerobot_dataset_factory(root=root, features=features, repo_id=REPO_ID)
    for ep in range(4):
        for _ in range(5):
            dataset.add_frame(
                {
                    "action": np.full(2, ep, dtype=np.float32),
                    "observation.state": np.zeros(2, dtype=np.float32),
                    "task": "t",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    (root / "meta" / "provenance.json").write_text('{"sim": "abc"}')
    return root


def test_marking_accumulates_atomically(root):
    assert read_deleted_episodes(root) == []
    assert mark_episode_deleted(root, 2) == [2]
    assert mark_episode_deleted(root, 0) == [0, 2]
    assert mark_episode_deleted(root, 2) == [0, 2]
    assert json.loads((root / DELETED_EPISODES_FILE).read_text()) == {"episodes": [0, 2]}
    assert not list((root / "meta").glob("*.tmp"))


def test_apply_removes_the_marked_episodes_and_keeps_provenance(root):
    mark_episode_deleted(root, 1)
    mark_episode_deleted(root, 3)
    assert apply_deleted_episodes(root, REPO_ID)
    dataset = LeRobotDataset(REPO_ID, root=root)
    assert dataset.num_episodes == 2 and dataset.num_frames == 10
    assert sorted({float(a[0]) for a in dataset.hf_dataset["action"]}) == [0.0, 2.0]
    assert (root / "meta" / "provenance.json").read_text() == '{"sim": "abc"}'
    assert not (root / DELETED_EPISODES_FILE).exists()
    assert not list(root.parent.glob("dataset.*-delete")), "temporary copies are removed"


def test_apply_without_tombstones_is_a_noop(root):
    assert apply_deleted_episodes(root, REPO_ID)
    assert LeRobotDataset(REPO_ID, root=root).num_episodes == 4


def test_a_failed_rewrite_leaves_the_dataset_and_tombstone(root, monkeypatch, caplog):
    mark_episode_deleted(root, 1)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("lerobot.datasets.dataset_tools.delete_episodes", boom)
    assert not apply_deleted_episodes(root, REPO_ID)
    assert read_deleted_episodes(root) == [1]
    assert LeRobotDataset(REPO_ID, root=root).num_episodes == 4
    assert "apply_deleted_episodes.py" in caplog.text
    assert not list(root.parent.glob("dataset.*-delete"))


def test_a_failed_swap_restores_the_original(root, monkeypatch):
    mark_episode_deleted(root, 1)
    real_rename = Path.rename

    def rename(self, target):
        if str(self).endswith(".tmp-delete"):
            raise OSError("swap failed")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    assert not apply_deleted_episodes(root, REPO_ID)
    monkeypatch.undo()
    assert read_deleted_episodes(root) == [1]
    assert LeRobotDataset(REPO_ID, root=root).num_episodes == 4


def test_deleting_every_episode_keeps_the_dataset(root, caplog):
    for ep in range(4):
        mark_episode_deleted(root, ep)
    assert not apply_deleted_episodes(root, REPO_ID)
    assert read_deleted_episodes(root) == [0, 1, 2, 3]
    assert LeRobotDataset(REPO_ID, root=root).num_episodes == 4
    assert "Every episode" in caplog.text


def test_a_crash_between_the_renames_is_recovered(root):
    mark_episode_deleted(root, 0)
    aside = root.with_name(root.name + deleted_episodes._ASIDE_SUFFIX)
    root.rename(aside)
    assert apply_deleted_episodes(root, REPO_ID)
    assert LeRobotDataset(REPO_ID, root=root).num_episodes == 3
    assert not aside.exists()
