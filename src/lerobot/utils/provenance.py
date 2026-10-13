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

"""Record which code produced a dataset (git commits, Hub snapshot revisions)."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import lerobot

logger = logging.getLogger(__name__)

PROVENANCE_FILE = "meta/provenance.json"


def git_info(path: str | Path) -> dict[str, Any] | None:
    """``{"commit", "dirty"}`` of the git checkout containing ``path``, or None (no git, not a repo)."""
    path = Path(path)
    cwd = path if path.is_dir() else path.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": bool(status.strip())}


def hub_snapshot_info(path: str | Path) -> dict[str, str] | None:
    """``{"repo_id", "revision"}`` for a path inside a Hub cache snapshot
    (``.../<models|datasets|spaces>--owner--name/snapshots/<revision>/...``), else None."""
    parts = Path(path).parts
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] == "snapshots" and parts[i - 1].count("--") >= 2:
            kind, _, repo = parts[i - 1].partition("--")
            if kind in ("models", "datasets", "spaces"):
                return {"repo_id": repo.replace("--", "/"), "revision": parts[i + 1]}
    return None


def code_info(path: str | Path) -> dict[str, Any]:
    """Where code came from: a Hub snapshot (repo_id + revision) or a checkout (path + git state)."""
    snapshot = hub_snapshot_info(path)
    if snapshot is not None:
        return snapshot
    return {"path": str(path), **(git_info(path) or {})}


def lerobot_info() -> dict[str, Any]:
    """Git commit and dirty flag of the installed LeRobot checkout (None fields when it is not a git repo)."""
    return git_info(Path(lerobot.__file__).parent) or {"commit": None, "dirty": None}


def write_provenance(robot: Any, root: str | Path) -> Path | None:
    """Write `robot.provenance()` to `<root>/meta/provenance.json` (overwrites). No-op without the hook."""
    provenance = getattr(robot, "provenance", None)
    if not callable(provenance):
        return None
    out = Path(root) / PROVENANCE_FILE
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(provenance(), indent=2) + "\n")
    except Exception as e:  # noqa: BLE001 - never block a recording over metadata
        logger.warning(f"Could not write {out}: {e}")
        return None
    logger.info(f"Wrote {out}")
    return out
