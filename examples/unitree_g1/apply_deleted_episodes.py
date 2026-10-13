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

"""Remove the episodes the operator deleted (Quest B button) from a recorded dataset.

`lerobot-record` normally does this when it finishes. If it crashed first, the dataset root still has
`meta/deleted_episodes.json`; this rewrites the dataset without those episodes.

    python examples/unitree_g1/apply_deleted_episodes.py --root <dataset root> [--repo-id <repo id>]
"""

import argparse
import logging
import sys

from lerobot.datasets.deleted_episodes import apply_deleted_episodes, read_deleted_episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--repo-id", default=None, help="Dataset repo id (default: local/<root name>)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(f"Deleted episodes marked in {args.root}: {read_deleted_episodes(args.root)}")
    return 0 if apply_deleted_episodes(args.root, args.repo_id) else 1


if __name__ == "__main__":
    sys.exit(main())
