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

import json
import shutil
import subprocess

import pytest

from lerobot.utils.provenance import code_info, git_info, hub_snapshot_info, write_provenance

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def test_hub_snapshot_info_parses_cache_paths(tmp_path):
    snap = tmp_path / "hub" / "models--k-valentin--unitree-g1-mujoco" / "snapshots" / "abc123"
    assert hub_snapshot_info(snap) == {"repo_id": "k-valentin/unitree-g1-mujoco", "revision": "abc123"}
    # Deeper paths inside the snapshot resolve to the same revision
    assert hub_snapshot_info(snap / "worlds" / "w" / "world.xml")["revision"] == "abc123"


def test_hub_snapshot_info_dataset_repo_and_non_cache(tmp_path):
    assert hub_snapshot_info(tmp_path / "datasets--o--n" / "snapshots" / "r")["repo_id"] == "o/n"
    assert hub_snapshot_info(tmp_path / "checkout" / "sim") is None
    assert hub_snapshot_info(tmp_path / "models--o--n" / "blobs" / "x") is None


@needs_git
def test_git_info_commit_and_dirty(tmp_path):
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    run("init")
    (tmp_path / "f.txt").write_text("a")
    run("add", "f.txt")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert git_info(tmp_path) == {"commit": head, "dirty": False}
    (tmp_path / "f.txt").write_text("b")
    assert git_info(tmp_path / "f.txt") == {"commit": head, "dirty": True}
    assert code_info(tmp_path) == {"path": str(tmp_path), "commit": head, "dirty": True}


def test_git_info_none_outside_repo_or_without_git(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_info(tmp_path) is None
    assert code_info(tmp_path) == {"path": str(tmp_path)}
    monkeypatch.setenv("PATH", "")
    assert git_info(tmp_path) is None


def test_write_provenance_writes_json_and_overwrites(tmp_path):
    class Robot:
        value = 1

        def provenance(self):
            return {"v": self.value}

    robot = Robot()
    out = write_provenance(robot, tmp_path)
    assert out == tmp_path / "meta" / "provenance.json"
    assert json.loads(out.read_text()) == {"v": 1}
    robot.value = 2
    write_provenance(robot, tmp_path)
    assert json.loads(out.read_text()) == {"v": 2}


def test_write_provenance_noop_without_hook(tmp_path):
    assert write_provenance(object(), tmp_path) is None
    assert not (tmp_path / "meta").exists()
