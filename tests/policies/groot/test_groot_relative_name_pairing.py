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

"""GR00T N1.7 relative actions with relative_action_pairing='name' (action and state in different orders)."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.processor_groot import (
    N1_7_NATIVE_ACTION_HORIZON,
    GrootN17ActionDecodeStep,
    GrootN17PackInputsStep,
    GrootReorderStateStep,
    _name_pairing_plan,
    make_groot_pre_post_processors,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.groot.test_groot_n1_7 import _groot_features, _native_action_chunk

# A G1-controller-like layout: legs/waist first in the state, arms first in the action, plus commands
# (remote axis, base height) with no state counterpart and a hand closure kept absolute.
STATE_NAMES = [
    "kLeftHipPitch.q",
    "kLeftKnee.q",
    "kWaistYaw.q",
    "kLeftShoulderPitch.q",
    "kLeftElbow.q",
    "kHeadYaw.q",
    "kLeftHand.closure",
]
ACTION_NAMES = [
    "kLeftShoulderPitch.q",
    "kLeftElbow.q",
    "remote.lx",
    "kHeadYaw.q",
    "kLeftHand.closure",
    "kBaseHeight.cmd",
]
EXPECTED_PERMUTATION = [3, 4, 0, 5, 1, 2, 6]


def test_name_pairing_plan_pairs_by_name_and_keeps_commands_absolute():
    permutation, exclude = _name_pairing_plan(ACTION_NAMES, STATE_NAMES, ["closure"])
    assert permutation == EXPECTED_PERMUTATION
    assert sorted(permutation) == list(range(len(STATE_NAMES)))
    for action_index, name in enumerate(ACTION_NAMES):
        if name in ("kLeftShoulderPitch.q", "kLeftElbow.q", "kHeadYaw.q"):
            assert STATE_NAMES[permutation[action_index]] == name
    assert exclude == ["closure", "remote.lx", "kBaseHeight.cmd"]


def test_name_pairing_plan_matches_index_pairing_for_aligned_layouts():
    names = ["a.pos", "b.pos", "gripper.pos"]
    permutation, exclude = _name_pairing_plan(names, names, ["gripper"])
    assert permutation == [0, 1, 2]
    assert exclude == ["gripper"]


@pytest.mark.parametrize(
    "action_names, state_names, message",
    [
        (["a", "b", "c"], ["a", "b"], "at least as many"),
        (["a"], ["a", "a"], "unique"),
        (["x"], ["a", "b"], "no action feature"),
        (["kLeftElbow", "kLeftElbow.q"], ["kLeftElbow.q", "b"], "substring"),
    ],
)
def test_name_pairing_plan_errors(action_names, state_names, message):
    with pytest.raises(ValueError, match=message):
        _name_pairing_plan(action_names, state_names, [])


def test_reorder_state_step_applies_and_round_trips(tmp_path):
    step = GrootReorderStateStep(indices=EXPECTED_PERMUTATION)
    state = torch.arange(7, dtype=torch.float32).unsqueeze(0)
    out = step({TransitionKey.OBSERVATION: {OBS_STATE: state}})
    torch.testing.assert_close(out[TransitionKey.OBSERVATION][OBS_STATE], state[:, EXPECTED_PERMUTATION])

    pipeline = PolicyProcessorPipeline(steps=[step], name="policy_preprocessor")
    pipeline.save_pretrained(tmp_path)
    loaded = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    assert isinstance(loaded.steps[0], GrootReorderStateStep)
    assert loaded.steps[0].indices == EXPECTED_PERMUTATION


def test_config_validates_pairing():
    input_features, output_features = _groot_features(state_dim=7, action_dim=6)
    config = GrootConfig(input_features=input_features, output_features=output_features, device="cpu")
    assert config.relative_action_pairing == "index"
    with pytest.raises(ValueError, match="relative_action_pairing"):
        GrootConfig(
            input_features=input_features,
            output_features=output_features,
            device="cpu",
            relative_action_pairing="by_name",
        )


def _state(hip, knee, waist, shoulder, elbow, head, closure):
    return torch.tensor([hip, knee, waist, shoulder, elbow, head, closure])


def _action_rows(state, delta, remote, closure, height):
    shoulder, elbow, head = state[3].item(), state[4].item(), state[5].item()
    return [shoulder + delta, elbow + delta, remote, head + delta / 2, closure, height]


def _build_name_paired_processors(tmp_path, monkeypatch, pairing="name"):
    input_features, output_features = _groot_features(state_dim=7, action_dim=6)
    config = GrootConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_bf16=False,
        action_decode_transform=None,
        use_relative_actions=True,
        relative_exclude_joints=["closure"],
        relative_action_pairing=pairing,
    )
    states = [_state(1.0, 2.0, 0.5, 0.3, 1.2, 0.1, 0.4), _state(-1.0, 1.5, -0.5, 0.6, 0.8, -0.2, 0.9)]
    samples = [
        {
            OBS_STATE: states[0],
            ACTION: _native_action_chunk(
                [_action_rows(states[0], 0.02, 0.5, 0.9, 0.74), _action_rows(states[0], 0.04, 0.6, 1.0, 0.70)]
            ),
        },
        {
            OBS_STATE: states[1],
            ACTION: _native_action_chunk(
                [
                    _action_rows(states[1], 0.01, -0.5, 0.1, 0.60),
                    _action_rows(states[1], 0.03, -0.4, 0.0, 0.55),
                ]
            ),
        },
    ]
    stacked_states = torch.stack(states)
    all_actions = torch.cat([sample[ACTION] for sample in samples])
    absolute_stats = {
        OBS_STATE: {"min": stacked_states.min(0).values, "max": stacked_states.max(0).values},
        ACTION: {"min": all_actions.min(0).values, "max": all_actions.max(0).values},
    }
    runtime_meta = SimpleNamespace(
        repo_id="local/name_pairing",
        root=tmp_path,
        revision="main",
        fps=30,
        stats=absolute_stats,
        features={ACTION: {"names": ACTION_NAMES}, OBS_STATE: {"names": STATE_NAMES}},
    )

    class _Dataset:
        meta = runtime_meta

        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            return samples[idx]

    monkeypatch.setattr("lerobot.policies.groot.processor_groot.LeRobotDataset", lambda *a, **k: _Dataset())
    config._runtime_dataset_meta = runtime_meta
    preprocessor, postprocessor = make_groot_pre_post_processors(config, dataset_stats=absolute_stats)
    return preprocessor, postprocessor, absolute_stats


def test_name_paired_processors_pair_arm_and_head_with_their_own_state(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    preprocessor, postprocessor, absolute_stats = _build_name_paired_processors(tmp_path, monkeypatch)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    steps = json.loads((tmp_path / "policy_preprocessor.json").read_text())["steps"]
    registry_names = [step.get("registry_name") for step in steps]
    assert registry_names.index("groot_reorder_state") < registry_names.index("groot_n1_7_pack_inputs_v1")
    assert steps[registry_names.index("groot_reorder_state")]["config"]["indices"] == EXPECTED_PERMUTATION

    pack_entry = steps[registry_names.index("groot_n1_7_pack_inputs_v1")]
    action_cfg = pack_entry["config"]["modality_config"]["action"]
    reps = [cfg["rep"] for cfg in action_cfg["action_configs"]]
    # arm group, remote.lx, head group, closure, kBaseHeight.cmd
    assert reps == ["RELATIVE", "ABSOLUTE", "RELATIVE", "ABSOLUTE", "ABSOLUTE"]
    arm_key, head_key = action_cfg["modality_keys"][0], action_cfg["modality_keys"][2]

    relative = pack_entry["config"]["raw_stats"]["relative_action"]
    # Deltas against the same-named state (0.01..0.04), never shoulder - hip (about -0.7 / +1.6).
    assert relative[arm_key]["min"][0] == pytest.approx([0.01, 0.01])
    assert relative[arm_key]["max"][0] == pytest.approx([0.02, 0.02])
    assert relative[head_key]["min"][1] == pytest.approx([0.015])
    assert len(relative[arm_key]["min"]) == N1_7_NATIVE_ACTION_HORIZON

    state_keys = pack_entry["config"]["modality_config"]["state"]["modality_keys"]
    assert state_keys[-1] == "state_extra"
    pack_state = load_file(tmp_path / pack_entry["state_file"])
    permuted_min = absolute_stats[OBS_STATE]["min"][EXPECTED_PERMUTATION]
    torch.testing.assert_close(pack_state[f"{OBS_STATE}.min"], permuted_min)
    assert pack_state[f"{OBS_STATE}.min"].shape == (len(STATE_NAMES),)


def test_name_paired_processors_decode_back_to_absolute_after_reload(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    preprocessor, postprocessor, _ = _build_name_paired_processors(tmp_path, monkeypatch)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    loaded_pre = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    loaded_post = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename="policy_postprocessor.json"
    )

    reorder = next(step for step in loaded_pre.steps if isinstance(step, GrootReorderStateStep))
    pack = next(step for step in loaded_pre.steps if isinstance(step, GrootN17PackInputsStep))
    decode = next(step for step in loaded_post.steps if isinstance(step, GrootN17ActionDecodeStep))
    decode.pack_step = pack

    state = _state(1.0, 2.0, 0.5, 0.3, 1.2, 0.1, 0.4).unsqueeze(0)
    transition = reorder(
        {TransitionKey.OBSERVATION: {OBS_STATE: state}, TransitionKey.COMPLEMENTARY_DATA: {}}
    )
    pack(transition)

    action_cfg = pack.modality_config["action"]
    arm_key, head_key = action_cfg["modality_keys"][0], action_cfg["modality_keys"][2]
    arm_rel = pack.raw_stats["relative_action"][arm_key]
    head_rel = pack.raw_stats["relative_action"][head_key]
    decoded = decode({TransitionKey.ACTION: torch.zeros(1, N1_7_NATIVE_ACTION_HORIZON, 6)})[
        TransitionKey.ACTION
    ]

    arm_mid = (torch.tensor(arm_rel["min"][0]) + torch.tensor(arm_rel["max"][0])) / 2
    head_mid = (torch.tensor(head_rel["min"][0]) + torch.tensor(head_rel["max"][0])) / 2
    # Relative columns come back as own-state + offset (shoulder 0.3, elbow 1.2, head 0.1), in action order.
    torch.testing.assert_close(decoded[0, 0, :2], torch.tensor([0.3, 1.2]) + arm_mid)
    torch.testing.assert_close(decoded[0, 0, 3:4], torch.tensor([0.1]) + head_mid)
    assert decoded.shape[-1] == len(ACTION_NAMES)


def test_index_pairing_on_this_layout_subtracts_the_wrong_state(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    preprocessor, _, _ = _build_name_paired_processors(tmp_path, monkeypatch, pairing="index")
    assert not any(isinstance(step, GrootReorderStateStep) for step in preprocessor.steps)
    pack = next(step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep))
    first_group = pack.modality_config["action"]["modality_keys"][0]
    shoulder_min = pack.raw_stats["relative_action"][first_group]["min"][0][0]
    # Shoulder target minus hip state (0.32 - 1.0), which name pairing avoids.
    assert shoulder_min == pytest.approx(-0.68)


def test_name_pairing_rejects_missing_state_names(tmp_path, monkeypatch):
    input_features, output_features = _groot_features(state_dim=7, action_dim=6)
    config = GrootConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_relative_actions=True,
        relative_action_pairing="name",
    )
    meta = SimpleNamespace(stats={}, features={ACTION: {"names": ACTION_NAMES}})
    with pytest.raises(ValueError, match="feature names"):
        make_groot_pre_post_processors(config, dataset_stats={}, dataset_meta=meta)
