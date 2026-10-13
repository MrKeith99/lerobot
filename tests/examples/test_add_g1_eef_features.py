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

"""Flange / rot6d math of examples/unitree_g1/add_g1_eef_features.py (fake FK, plus a real-FK check)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples/unitree_g1/add_g1_eef_features.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("add_g1_eef_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose(rot: np.ndarray, pos) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


def test_flange_is_tcp_moved_back_along_its_own_x(script):
    tcp = _pose(_rot_z(np.pi / 2), [0.3, 0.1, 0.2])  # tcp x axis points along world +y
    flange = script.flange_pose(tcp, 0.05)
    np.testing.assert_allclose(flange[:3, 3], [0.3, 0.05, 0.2], atol=1e-12)
    np.testing.assert_allclose(flange[:3, :3], tcp[:3, :3])


def test_eef_features_layout_rot6d_rows_and_joint_selection(script):
    rot_left, rot_right = _rot_z(0.3), _rot_z(-0.7)

    def fake_forward(q_arm):
        assert q_arm.shape == (2,)  # only the selected joints reach FK, in FK order
        return {
            "left": _pose(rot_left, [q_arm[0] + 0.05, 0.1, 0.2]),
            "right": _pose(rot_right, [q_arm[1] + 0.05, -0.1, 0.2]),
        }

    q = np.array([[9.0, 1.0, 9.0, 2.0], [9.0, 3.0, 9.0, 4.0]])
    out = script.eef_features(fake_forward, np.array([3, 1]), q, {"left": 0.05, "right": 0.05})

    assert out.shape == (2, 18) and out.dtype == np.float32
    for i, (qa, qb) in enumerate([(2.0, 1.0), (4.0, 3.0)]):
        left_tcp = np.array([qa + 0.05, 0.1, 0.2])
        right_tcp = np.array([qb + 0.05, -0.1, 0.2])
        np.testing.assert_allclose(out[i, :3], left_tcp - 0.05 * rot_left[:, 0], atol=1e-6)
        np.testing.assert_allclose(out[i, 3:9], np.concatenate([rot_left[0], rot_left[1]]), atol=1e-6)
        np.testing.assert_allclose(out[i, 9:12], right_tcp - 0.05 * rot_right[:, 0], atol=1e-6)
        np.testing.assert_allclose(out[i, 12:18], np.concatenate([rot_right[0], rot_right[1]]), atol=1e-6)


def test_names_and_arm_key(script):
    assert len(script.EEF_NAMES) == 18
    assert script.EEF_NAMES[:3] == ["kLeftEef.x", "kLeftEef.y", "kLeftEef.z"]
    assert script.EEF_NAMES[9] == "kRightEef.x" and script.EEF_NAMES[-1] == "kRightEef.r5"
    assert script.arm_key("left_shoulder_pitch_joint") == "kLeftShoulderPitch.q"


def test_flange_matches_mount_frame_with_real_fk(script):
    pytest.importorskip("pinocchio")
    pytest.importorskip("casadi")
    xr = pytest.importorskip("xr_teleoperate.core")
    try:
        arm_ik = xr.G1ArmIK("23dof", "amazing_hand")
    except Exception as e:  # noqa: BLE001 - sim repo not available offline
        pytest.skip(f"cannot build G1ArmIK: {e}")
    import pinocchio as pin

    offsets = script.fk_tcp_offsets(arm_ik)
    q = np.random.default_rng(0).uniform(-0.5, 0.5, arm_ik.nq)
    poses = arm_ik.forward(q)
    pin.framesForwardKinematics(arm_ik.model, arm_ik.data, q)
    for side in ("left", "right"):
        mount = arm_ik.data.oMf[arm_ik.model.getFrameId(f"{side}_ee_mount")].homogeneous
        np.testing.assert_allclose(script.flange_pose(poses[side], offsets[side]), mount, atol=1e-9)
