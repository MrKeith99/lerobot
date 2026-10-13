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

"""GR00T N1.7 relative end-effector (SE(3), xyz+rot6d) actions via relative_eef_prefixes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import (
    N1_7_NATIVE_ACTION_HORIZON,
    GrootN17ActionDecodeStep,
    GrootN17PackInputsStep,
    GrootReorderStateStep,
    _infer_n1_7_action_groups,
    _make_relative_action_training_stats,
    _name_pairing_plan,
    make_groot_pre_post_processors,
)
from lerobot.policies.groot.utils import (
    absolute_eef_to_relative_torch,
    find_eef_blocks,
    homogeneous_to_xyz_rot6d,
    relative_eef_to_absolute,
    xyz_rot6d_to_homogeneous,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.groot.test_groot_n1_7 import _groot_features

H = N1_7_NATIVE_ACTION_HORIZON
EEF = ["x", "y", "z", "r0", "r1", "r2", "r3", "r4", "r5"]
LEFT = [f"kLeftEef.{s}" for s in EEF]
RIGHT = [f"kRightEef.{s}" for s in EEF]
ACTION_NAMES = ["kArmA.q", "kArmB.q", "kLeftHand.closure", "kNav.cmd", *LEFT, *RIGHT]
STATE_NAMES = [*RIGHT, "kArmB.q", "kArmA.q", *LEFT, "kLeftHand.closure"]
PREFIXES = ["kLeftEef", "kRightEef"]


def _random_poses(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        transform = np.eye(4)
        transform[:3, :3] = q
        transform[:3, 3] = rng.uniform(-1, 1, size=3)
        poses.append(homogeneous_to_xyz_rot6d(transform))
    return torch.tensor(np.stack(poses), dtype=torch.float32)


def _rigid(seed: int) -> np.ndarray:
    return xyz_rot6d_to_homogeneous(_random_poses(1, seed)[0].numpy())


def _apply(g: np.ndarray, poses: torch.Tensor) -> torch.Tensor:
    flat = poses.reshape(-1, 9).numpy()
    out = [homogeneous_to_xyz_rot6d(g @ xyz_rot6d_to_homogeneous(p)) for p in flat]
    return torch.tensor(np.stack(out), dtype=torch.float32).reshape(poses.shape)


def test_round_trip_batched_chunk():
    state = _random_poses(4, seed=1)
    action = _random_poses(4 * 7, seed=2).reshape(4, 7, 9)
    relative = absolute_eef_to_relative_torch(action, state)
    assert relative.shape == (4, 7, 9)
    decoded = relative_eef_to_absolute(relative.numpy(), state.numpy())
    np.testing.assert_allclose(decoded, action.numpy(), atol=1e-5)


def test_identity_action_equals_state():
    state = _random_poses(3, seed=3)
    relative = absolute_eef_to_relative_torch(state[:, None, :].repeat(1, 5, 1), state)
    expected = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32).expand_as(relative)
    torch.testing.assert_close(relative, expected, atol=1e-5, rtol=0)


def test_frame_invariance():
    state = _random_poses(2, seed=4)
    action = _random_poses(2 * 5, seed=5).reshape(2, 5, 9)
    g = _rigid(6)
    base = absolute_eef_to_relative_torch(action, state)
    moved = absolute_eef_to_relative_torch(_apply(g, action), _apply(g, state))
    torch.testing.assert_close(moved, base, atol=1e-5, rtol=0)


def test_find_eef_blocks_errors():
    assert find_eef_blocks(ACTION_NAMES, PREFIXES) == {"kLeftEef": 4, "kRightEef": 13}
    with pytest.raises(ValueError, match="no 'kHeadEef.x'"):
        find_eef_blocks(ACTION_NAMES, ["kHeadEef"])
    swapped = [*ACTION_NAMES]
    swapped[4], swapped[5] = swapped[5], swapped[4]
    with pytest.raises(ValueError, match="contiguous columns"):
        find_eef_blocks(swapped, PREFIXES)
    with pytest.raises(ValueError, match="contiguous columns"):
        find_eef_blocks(ACTION_NAMES[:-1], ["kRightEef"])


def test_group_inference_with_mixed_names():
    groups = _infer_n1_7_action_groups(
        ACTION_NAMES, action_dim=len(ACTION_NAMES), exclude_joints=["closure", "cmd"], eef_prefixes=PREFIXES
    )
    assert [(g.key, g.relative, g.eef) for g in groups] == [
        ("single_arm", True, False),
        ("closure", False, False),
        ("cmd", False, False),
        ("kLeftEef", True, True),
        ("kRightEef", True, True),
    ]
    assert groups[3].indices == list(range(4, 13))
    assert groups[4].indices == list(range(13, 22))
    # Without prefixes the EEF columns are plain joint-wise relative columns (unchanged behaviour).
    plain = _infer_n1_7_action_groups(ACTION_NAMES, action_dim=22, exclude_joints=["closure", "cmd"])
    assert [g.eef for g in plain] == [False] * 3 + [False]


def _config(**kwargs):
    input_features, output_features = _groot_features(state_dim=21, action_dim=22)
    return GrootConfig(input_features=input_features, output_features=output_features, device="cpu", **kwargs)


def test_config_validation():
    ok = {"use_relative_actions": True, "relative_action_pairing": "name"}
    assert _config(**ok, relative_eef_prefixes=PREFIXES).relative_eef_prefixes == PREFIXES
    with pytest.raises(ValueError, match="requires use_relative_actions"):
        _config(relative_eef_prefixes=PREFIXES)
    with pytest.raises(ValueError, match="requires use_relative_actions"):
        _config(use_relative_actions=True, relative_eef_prefixes=PREFIXES)
    with pytest.raises(ValueError, match="relative_exclude_joints"):
        _config(**ok, relative_eef_prefixes=PREFIXES, relative_exclude_joints=["Eef"])
    with pytest.raises(ValueError, match="duplicates"):
        _config(**ok, relative_eef_prefixes=["kLeftEef", "kLeftEef"])


def _dataset(tmp_path):
    states, actions = [], []
    for i in range(3):
        left_s, right_s = _random_poses(1, seed=20 + i)[0], _random_poses(1, seed=30 + i)[0]
        state = torch.zeros(21)
        state[0:9], state[11:20] = right_s, left_s
        state[9], state[10], state[20] = 0.5 + i, 0.2, 0.1
        chunk = torch.zeros(N1_7_NATIVE_ACTION_HORIZON, 22)
        chunk[:, 0], chunk[:, 1], chunk[:, 2], chunk[:, 3] = 0.2, 0.5 + i, 0.3, 0.1 * i
        chunk[:, 4:13] = left_s + 0.01 * torch.arange(N1_7_NATIVE_ACTION_HORIZON)[:, None] * torch.tensor(
            [1, 0, 0, 0, 0, 0, 0, 0, 0.0]
        )
        chunk[:, 13:22] = right_s
        states.append(state)
        actions.append(chunk)
    return states, actions


def _build(tmp_path, monkeypatch, states, actions):
    config = _config(
        use_bf16=False,
        action_decode_transform=None,
        use_relative_actions=True,
        relative_exclude_joints=["closure", "cmd"],
        relative_action_pairing="name",
        relative_eef_prefixes=PREFIXES,
    )
    all_states, all_actions = torch.stack(states), torch.cat(actions)
    stats = {
        OBS_STATE: {"min": all_states.min(0).values, "max": all_states.max(0).values},
        ACTION: {"min": all_actions.min(0).values, "max": all_actions.max(0).values},
    }
    meta = SimpleNamespace(
        repo_id="local/eef",
        root=tmp_path,
        revision="main",
        fps=30,
        stats=stats,
        features={ACTION: {"names": ACTION_NAMES}, OBS_STATE: {"names": STATE_NAMES}},
    )

    class _Dataset:
        def __init__(self):
            self.meta = meta

        def __len__(self):
            return len(states)

        def __getitem__(self, idx):
            return {ACTION: actions[idx], OBS_STATE: states[idx]}

    monkeypatch.setattr("lerobot.policies.groot.processor_groot.LeRobotDataset", lambda *a, **k: _Dataset())
    config._runtime_dataset_meta = meta
    return (*make_groot_pre_post_processors(config, dataset_stats=stats), stats)


def test_relative_stats_use_se3_deltas(tmp_path):
    pytest.importorskip("datasets")
    states, actions = _dataset(tmp_path)
    permutation, exclude = _name_pairing_plan(ACTION_NAMES, STATE_NAMES, ["closure", "cmd"])

    class _Dataset:
        meta = SimpleNamespace(stats={})

        def __len__(self):
            return len(states)

        def __getitem__(self, idx):
            return {ACTION: actions[idx], OBS_STATE: states[idx]}

    stats = _make_relative_action_training_stats(
        _Dataset(),
        exclude_joints=exclude,
        action_names=ACTION_NAMES,
        state_permutation=permutation,
        eef_prefixes=PREFIXES,
    )[ACTION]
    assert np.asarray(stats["min"]).shape == (N1_7_NATIVE_ACTION_HORIZON, 22)
    # Right EEF never moves from the state pose: identity delta at every step (not a column subtraction).
    identity = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(stats["min"])[:, 13:22], np.tile(identity, (H, 1)), atol=1e-5)
    np.testing.assert_allclose(np.asarray(stats["max"])[:, 13:22], np.tile(identity, (H, 1)), atol=1e-5)
    # Left EEF is translated by 0.01 t along world x with its orientation unchanged, so in the state
    # frame the delta is a rotated vector of norm 0.01 t (components bounded by it), rotation identity.
    lo, hi = np.asarray(stats["min"])[:, 4:13], np.asarray(stats["max"])[:, 4:13]
    np.testing.assert_allclose(lo[:, 3:], np.tile(identity[3:], (H, 1)), atol=1e-5)
    np.testing.assert_allclose(hi[:, 3:], np.tile(identity[3:], (H, 1)), atol=1e-5)
    for t in range(1, N1_7_NATIVE_ACTION_HORIZON):
        assert np.abs(np.concatenate([lo[t, :3], hi[t, :3]])).max() <= 0.01 * t + 1e-5
        assert np.abs(hi[t, :3] - lo[t, :3]).max() > 0
    # Joints are still plain differences against the same-named state (kArmA: 0.2 - 0.2).
    assert np.asarray(stats["min"])[0, 0] == pytest.approx(0.2 - 0.2)
    # Excluded closure/cmd stay absolute.
    assert np.asarray(stats["min"])[0, 2] == pytest.approx(0.3)


def test_processors_keep_eef_group_config_and_decode_after_reload(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    states, actions = _dataset(tmp_path)
    preprocessor, postprocessor, _ = _build(tmp_path, monkeypatch, states, actions)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    steps = json.loads((tmp_path / "policy_preprocessor.json").read_text())["steps"]
    pack_entry = next(s for s in steps if s.get("registry_name") == "groot_n1_7_pack_inputs_v1")
    action_cfg = pack_entry["config"]["modality_config"]["action"]
    assert action_cfg["modality_keys"] == ["single_arm", "closure", "cmd", "kLeftEef", "kRightEef"]
    eef_cfgs = action_cfg["action_configs"][3:]
    assert all(c["rep"] == "RELATIVE" and c["type"] == "EEF" and c["format"] == "XYZ_ROT6D" for c in eef_cfgs)
    assert [c["state_key"] for c in eef_cfgs] == ["kLeftEef", "kRightEef"]

    loaded_pre = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    loaded_post = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename="policy_postprocessor.json"
    )
    reorder = next(s for s in loaded_pre.steps if isinstance(s, GrootReorderStateStep))
    pack = next(s for s in loaded_pre.steps if isinstance(s, GrootN17PackInputsStep))
    decode = next(s for s in loaded_post.steps if isinstance(s, GrootN17ActionDecodeStep))
    decode.pack_step = pack

    state = states[0].unsqueeze(0)
    reordered = reorder({TransitionKey.OBSERVATION: {OBS_STATE: state}, TransitionKey.COMPLEMENTARY_DATA: {}})
    pack(reordered)

    # Training conversion: action == state pose -> identity relative pose.
    left_state, right_state = states[0][11:20], states[0][0:9]
    chunk = actions[0].unsqueeze(0).clone()
    chunk[:, :, 4:13] = left_state
    converted = pack._convert_relative_action_groups_for_training(
        chunk, reordered[TransitionKey.OBSERVATION][OBS_STATE]
    )
    identity = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32)
    torch.testing.assert_close(converted[0, :, 4:13], identity.expand(H, 9), atol=1e-5, rtol=0)
    torch.testing.assert_close(converted[0, :, 13:22], identity.expand(H, 9), atol=1e-5, rtol=0)

    # Inference decode: normalized 0 -> mid of the relative range, composed as T_state @ T_rel.
    decoded = decode({TransitionKey.ACTION: torch.zeros(1, N1_7_NATIVE_ACTION_HORIZON, 22)})[
        TransitionKey.ACTION
    ]
    rel = pack.raw_stats["relative_action"]["kLeftEef"]
    mid = ((np.asarray(rel["min"]) + np.asarray(rel["max"])) / 2).astype(np.float32)[None]
    expected = relative_eef_to_absolute(mid, left_state.numpy()[None])
    np.testing.assert_allclose(decoded[0, :, 4:13].numpy(), expected[0], atol=1e-4)
