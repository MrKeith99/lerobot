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

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

xr_core = pytest.importorskip("xr_teleoperate.core")

from lerobot.robots.unitree_g1.end_effectors import AMAZING_HAND, HAND_CLOSURE_KEYS  # noqa: E402
from lerobot.robots.unitree_g1.g1_utils import (  # noqa: E402
    BASE_HEIGHT_KEY,
    BODY_KEYS,
    GROOT_BASE_HEIGHT_DEFAULT,
    LOCOMOTION_TOGGLE_KEY,
    ROBOT_TYPE_FEEDBACK_KEY,
    SIM_RESET_KEY,
)
from lerobot.robots.unitree_g1.heads import HEAD_KEYS  # noqa: E402
from lerobot.teleoperators.unitree_g1_xr import (  # noqa: E402
    UnitreeG1XRTeleop,
    UnitreeG1XRTeleopConfig,
    unitree_g1_xr as xr_module,  # noqa: E402
)
from lerobot.teleoperators.unitree_g1_xr.unitree_g1_xr import arm_key  # noqa: E402
from lerobot.teleoperators.utils import TeleopEvents  # noqa: E402

XRFrame, HandInput = xr_core.XRFrame, xr_core.HandInput
ROBOT_TYPE = "unitree_g1-23dof_rev_1_0-amazing_hand-pan_tilt-d455"
ARM_23 = xr_core.arm_joint_names("23dof")
DT = 0.02


class FakeXR:
    def __init__(self):
        self.frame = XRFrame()
        self.recentered = 0
        self.rendered = []

    def render(self, image):
        self.rendered.append(image)

    def read(self):
        return self.frame

    def recenter(self):
        self.recentered += 1

    def close(self):
        pass


class FakeIK:
    def __init__(self, body="23dof"):
        self.joint_names = xr_core.arm_joint_names(body)
        self.solution = np.zeros(len(self.joint_names))
        self.reset_q = None

    def reset(self, q=None):
        self.reset_q = None if q is None else list(q)

    def solve(self, left, right, q_current=None):
        return SimpleNamespace(q=self.solution.copy())


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(xr_module, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    return now


def make_teleop(clock, **overrides):
    config = UnitreeG1XRTeleopConfig(**overrides)
    body = config.body
    teleop = UnitreeG1XRTeleop(config, xr_input=FakeXR(), arm_ik=FakeIK(body))
    teleop.connect()
    return teleop


def observation(**values):
    obs = dict.fromkeys(BODY_KEYS, 0.1)
    obs.update(dict.fromkeys(HEAD_KEYS, 0.0))
    obs.update(dict.fromkeys(HAND_CLOSURE_KEYS, 0.0))
    obs[ROBOT_TYPE_FEEDBACK_KEY] = ROBOT_TYPE
    obs.update(values)
    return obs


def step(teleop, clock, frame=None, n=1):
    if frame is not None:
        teleop.xr.frame = frame
    action = None
    for _ in range(n):
        clock[0] += DT
        action = teleop.get_action()
    return action


def press(teleop, clock, button, tracking=True):
    """One frame with the button down, one with it released."""
    step(teleop, clock, XRFrame(tracking=tracking, buttons={button: True}))
    return step(teleop, clock, XRFrame(tracking=tracking))


def test_arm_key_names():
    assert arm_key("left_shoulder_pitch_joint") == "kLeftShoulderPitch.q"
    assert arm_key("right_wrist_roll_joint") == "kRightWristRoll.q"
    assert all(arm_key(name) in BODY_KEYS for name in xr_core.arm_joint_names("29dof"))


def test_action_features_follow_the_embodiment(clock):
    features = make_teleop(clock).action_features
    assert set(HEAD_KEYS) <= set(features)
    assert set(HAND_CLOSURE_KEYS) <= set(features) and set(AMAZING_HAND.joint_keys()) <= set(features)
    bare = make_teleop(
        clock, body="29dof", end_effector="rubber_hand", head_mount="fixed", head_sensor="d435i"
    )
    assert not set(HEAD_KEYS) & set(bare.action_features)
    assert not set(HAND_CLOSURE_KEYS) & set(bare.action_features)
    assert set(step(bare, clock)) >= set(BODY_KEYS) | {BASE_HEIGHT_KEY}


def test_first_observation_sets_the_held_targets(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation(**{"kLeftElbow.q": 0.7, "kHeadYaw.q": 0.2, "kRightHand.closure": 0.5}))
    action = step(teleop, clock)
    assert action["kLeftElbow.q"] == pytest.approx(0.7)
    assert action["kLeftHipPitch.q"] == pytest.approx(0.1)
    assert action["kHeadYaw.q"] == pytest.approx(0.2)
    assert action["kRightHand.closure"] == pytest.approx(0.5)
    assert action[BASE_HEIGHT_KEY] == pytest.approx(GROOT_BASE_HEIGHT_DEFAULT)


@pytest.mark.parametrize(
    "robot_type",
    [
        "unitree_g1-29dof-amazing_hand-pan_tilt-d455",
        "unitree_g1-23dof_rev_1_0-dex3-pan_tilt-d455",
        "unitree_g1-23dof_rev_1_0-amazing_hand-pan_tilt-d435i",
    ],
)
def test_a_robot_of_another_embodiment_is_refused(clock, robot_type):
    with pytest.raises(ValueError, match="robot is"):
        make_teleop(clock).send_feedback(observation(**{ROBOT_TYPE_FEEDBACK_KEY: robot_type}))


def test_head_mount_is_checked_against_the_observation(clock):
    obs = observation()
    del obs[ROBOT_TYPE_FEEDBACK_KEY]
    for key in HEAD_KEYS:
        del obs[key]
    with pytest.raises(ValueError, match="pan/tilt"):
        make_teleop(clock).send_feedback(obs)


def test_nothing_moves_until_engaged(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation())
    teleop.arm_ik.solution[:] = 1.0
    frame = XRFrame(tracking=True, sticks={"left": (0.0, -1.0), "right": (0.5, 0.0)}, head_yaw=0.5)
    action = step(teleop, clock, frame, n=10)
    assert not teleop.engaged
    assert action["kLeftShoulderPitch.q"] == pytest.approx(0.1)
    assert action["kHeadYaw.q"] == 0.0
    assert action["kNavVx.cmd"] == 0.0 and action["remote.ly"] == 0.0


def test_engaging_recenters_and_starts_from_the_measured_arms(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation(**{"kRightElbow.q": 0.4}))
    press(teleop, clock, "a", tracking=False)
    assert not teleop.engaged, "engaging needs tracking"
    press(teleop, clock, "a")
    assert teleop.engaged and teleop.xr.recentered == 1
    assert teleop.arm_ik.reset_q[ARM_23.index("right_elbow_joint")] == pytest.approx(0.4)
    press(teleop, clock, "a")
    assert not teleop.engaged


def test_arm_targets_follow_the_ik_at_the_speed_limit(clock):
    teleop = make_teleop(clock, max_arm_speed_rad_s=2.0, engage_arm_speed_rad_s=2.0)
    teleop.send_feedback(observation())
    press(teleop, clock, "a")
    before = step(teleop, clock)["kLeftElbow.q"]
    teleop.arm_ik.solution[:] = 1.0
    after = step(teleop, clock)["kLeftElbow.q"]
    assert after - before == pytest.approx(2.0 * DT)
    action = step(teleop, clock, n=100)
    assert all(action[arm_key(name)] == pytest.approx(1.0) for name in ARM_23)
    assert action["kLeftWristPitch.q"] == pytest.approx(0.1), "no 23dof wrist pitch: held"


def test_targets_hold_while_tracking_is_lost_and_after_disengage(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation())
    teleop.arm_ik.solution[:] = 0.1
    press(teleop, clock, "a")
    teleop.arm_ik.solution[:] = 1.0
    held = step(
        teleop,
        clock,
        XRFrame(tracking=False, head_yaw=0.5, sticks={"left": (0.0, -1.0), "right": (0, 0)}),
        n=20,
    )
    assert teleop.engaged
    assert (
        held["kLeftElbow.q"] == pytest.approx(0.1) and held["kHeadYaw.q"] == 0.0 and held["kNavVx.cmd"] == 0.0
    )
    moved = step(teleop, clock, XRFrame(tracking=True), n=5)["kLeftElbow.q"]
    assert moved > 0.1
    press(teleop, clock, "a")
    teleop.arm_ik.solution[:] = -1.0
    assert step(teleop, clock, XRFrame(tracking=True), n=20)["kLeftElbow.q"] == pytest.approx(moved)
    assert not teleop.engaged


def test_head_follows_the_headset_within_its_limits(clock):
    teleop = make_teleop(clock, head_speed_rad_s=100.0)
    teleop.send_feedback(observation())
    press(teleop, clock, "a")
    action = step(teleop, clock, XRFrame(tracking=True, head_yaw=0.3, head_pitch=0.2))
    assert action["kHeadYaw.q"] == pytest.approx(0.3) and action["kHeadPitch.q"] == pytest.approx(0.2)
    action = step(teleop, clock, XRFrame(tracking=True, head_yaw=2.0, head_pitch=1.5))
    assert action["kHeadYaw.q"] == pytest.approx(0.7) and action["kHeadPitch.q"] == pytest.approx(0.8)
    slow = make_teleop(clock, head_speed_rad_s=1.0)
    slow.send_feedback(observation())
    press(slow, clock, "a")
    assert step(slow, clock, XRFrame(tracking=True, head_yaw=0.5))["kHeadYaw.q"] == pytest.approx(DT)


def test_triggers_close_the_hands(clock):
    teleop = make_teleop(clock, hand_blend_per_s=1000.0)
    teleop.send_feedback(observation())
    press(teleop, clock, "a")
    hands = {"left": HandInput(trigger=1.0), "right": HandInput(trigger=0.0)}
    action = step(teleop, clock, XRFrame(tracking=True, hands=hands))
    assert action["kLeftHand.closure"] == 1.0 and action["kRightHand.closure"] == 0.0
    assert tuple(action[k] for k in AMAZING_HAND.joint_keys("left")) == pytest.approx(
        AMAZING_HAND.closed_q["left"]
    )


def test_sticks_drive_navigation_and_buttons_the_base_height(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation())
    press(teleop, clock, "a")
    frame = XRFrame(tracking=True, sticks={"left": (0.0, -0.8), "right": (0.5, 0.0)}, buttons={"y": True})
    action = step(teleop, clock, frame, n=10)
    assert action["kNavVx.cmd"] == pytest.approx(0.8)
    assert action["kNavVy.cmd"] == 0.0
    assert action["kNavYawRate.cmd"] == pytest.approx(-0.5)
    assert action[BASE_HEIGHT_KEY] > GROOT_BASE_HEIGHT_DEFAULT and action["remote.button.0"] == 1.0
    small = XRFrame(tracking=True, sticks={"left": (0.05, 0.05), "right": (0.0, 0.0)})
    assert step(teleop, clock, small)["kNavVx.cmd"] == 0.0


def test_stick_clicks_end_the_episode(clock):
    teleop = make_teleop(clock)
    press(teleop, clock, "right_stick")
    events = teleop.get_teleop_events()
    assert events[TeleopEvents.SUCCESS] and events[TeleopEvents.TERMINATE_EPISODE]
    assert not teleop.get_teleop_events()[TeleopEvents.TERMINATE_EPISODE]
    press(teleop, clock, "left_stick")
    assert teleop.get_teleop_events()[TeleopEvents.RERECORD_EPISODE]


def test_hand_tracking_engages_when_tracking_starts(clock):
    teleop = make_teleop(clock, input_mode="hand")
    step(teleop, clock, XRFrame(tracking=False))
    assert not teleop.engaged
    step(teleop, clock, XRFrame(tracking=True))
    assert teleop.engaged


def test_unknown_display_mode_is_rejected():
    with pytest.raises(ValueError, match="display_mode"):
        UnitreeG1XRTeleopConfig(display_mode="stereo")


def test_head_camera_is_rendered_in_the_headset(clock):
    teleop = make_teleop(clock, display_mode="ego")
    image = np.zeros((480, 640, 3), np.uint8)
    teleop.send_feedback({"head_camera": image})
    teleop.send_feedback(observation() | {"head_camera": image})
    assert len(teleop.xr.rendered) == 2 and teleop.xr.rendered[0] is image


def test_pass_through_renders_nothing(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation() | {"head_camera": np.zeros((4, 4, 3), np.uint8)})
    assert teleop.xr.rendered == []


def test_missing_display_camera_warns_once_and_does_not_raise(clock, caplog):
    teleop = make_teleop(clock, display_mode="immersive")
    for _ in range(3):
        teleop.send_feedback(observation() | {"other": np.zeros((4, 4, 3), np.uint8)})
    assert teleop.xr.rendered == []
    warnings = [r for r in caplog.records if "head_camera" in r.getMessage()]
    assert len(warnings) == 1 and "other" in warnings[0].getMessage()


def test_render_errors_never_propagate(clock):
    teleop = make_teleop(clock, display_mode="ego")
    teleop.xr.render = lambda image: 1 / 0
    teleop.send_feedback({"head_camera": np.zeros((4, 4, 3), np.uint8)})


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[3] / "unitree-g1-mujoco" / "sim" / "mjcf" / "compose.py").is_file()
    and "XR_TELEOP_SIM_ROOT" not in os.environ,
    reason="needs a sim checkout with {side}_tcp sites",
)
def test_arms_reach_wrist_targets_with_the_composed_model_ik(clock):
    sim = os.environ.get("XR_TELEOP_SIM_ROOT", str(Path(__file__).resolve().parents[3] / "unitree-g1-mujoco"))
    ik = xr_core.G1ArmIK("23dof", "amazing_hand", sim_root=sim)
    teleop = UnitreeG1XRTeleop(UnitreeG1XRTeleopConfig(), xr_input=FakeXR(), arm_ik=ik)
    teleop.connect()
    teleop.send_feedback(observation(**dict.fromkeys(BODY_KEYS, 0.0)))
    q_goal = np.array([-0.3, 0.3, 0.1, 0.8, 0.2, -0.3, -0.3, -0.1, 0.8, -0.2])
    wrists = ik.forward(q_goal)
    press(teleop, clock, "a")
    action = step(teleop, clock, XRFrame(tracking=True, wrists=wrists), n=150)
    reached = ik.forward(np.array([action[arm_key(name)] for name in ik.joint_names]))
    for side in ("left", "right"):
        assert np.linalg.norm(reached[side][:3, 3] - wrists[side][:3, 3]) < 0.02


def test_arms_approach_the_operator_slowly_after_engaging(clock):
    teleop = make_teleop(clock, max_arm_speed_rad_s=3.0, engage_arm_speed_rad_s=0.5)
    teleop.send_feedback(observation())
    teleop.arm_ik.solution[:] = 1.0
    press(teleop, clock, "a")
    before = step(teleop, clock)["kLeftElbow.q"]
    assert step(teleop, clock)["kLeftElbow.q"] - before == pytest.approx(0.5 * DT)
    step(teleop, clock, n=200)
    teleop.arm_ik.solution[:] = 0.0
    before = step(teleop, clock)["kLeftElbow.q"]
    assert before - step(teleop, clock)["kLeftElbow.q"] == pytest.approx(3.0 * DT)


GRIPS = {"left_grip": True, "right_grip": True}


def grips_frame(**buttons):
    return XRFrame(tracking=True, buttons={**GRIPS, **buttons})


def toggles(teleop, clock, frame, n):
    return [step(teleop, clock, frame)[LOCOMOTION_TOGGLE_KEY] for _ in range(n)]


def test_band_toggle_key_is_a_zero_action_feature(clock):
    teleop = make_teleop(clock)
    assert LOCOMOTION_TOGGLE_KEY in teleop.action_features
    assert step(teleop, clock, XRFrame())[LOCOMOTION_TOGGLE_KEY] == 0.0
    assert step(teleop, clock, XRFrame(buttons={"left_grip": True}))[LOCOMOTION_TOGGLE_KEY] == 0.0


def test_band_toggle_hold_must_be_positive():
    with pytest.raises(ValueError, match="band_toggle_hold_s"):
        UnitreeG1XRTeleopConfig(band_toggle_hold_s=0.0)


def test_both_grips_fire_once_after_the_hold_time(clock):
    teleop = make_teleop(clock)
    values = toggles(teleop, clock, grips_frame(), 80)
    assert values[:49] == [0.0] * 49
    assert values.count(1.0) == 1
    assert values.index(1.0) in (49, 50)


def test_band_toggle_does_not_fire_before_the_hold_time(clock):
    teleop = make_teleop(clock, band_toggle_hold_s=2.0)
    assert toggles(teleop, clock, grips_frame(), 90) == [0.0] * 90


def test_band_toggle_needs_both_grips_continuously(clock):
    teleop = make_teleop(clock)
    assert toggles(teleop, clock, grips_frame(), 40) == [0.0] * 40
    step(teleop, clock, XRFrame(tracking=True, buttons={"left_grip": True}))
    assert toggles(teleop, clock, grips_frame(), 40) == [0.0] * 40


def test_band_toggle_never_fires_while_engaged(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation())
    press(teleop, clock, "a")
    assert teleop.engaged
    assert toggles(teleop, clock, grips_frame(), 100) == [0.0] * 100


def test_engaging_cancels_a_hold_in_progress(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation())
    assert toggles(teleop, clock, grips_frame(), 40) == [0.0] * 40
    assert toggles(teleop, clock, grips_frame(a=True), 1) == [0.0]
    assert teleop.engaged
    assert toggles(teleop, clock, grips_frame(), 100) == [0.0] * 100
    step(teleop, clock, grips_frame(a=True))
    assert not teleop.engaged
    assert toggles(teleop, clock, grips_frame(), 40) == [0.0] * 40


def test_band_toggle_rearms_after_release(clock):
    teleop = make_teleop(clock)
    assert toggles(teleop, clock, grips_frame(), 100).count(1.0) == 1
    step(teleop, clock, XRFrame(tracking=True))
    assert toggles(teleop, clock, grips_frame(), 100).count(1.0) == 1


def resets(teleop, clock, frame, n):
    return [step(teleop, clock, frame)[SIM_RESET_KEY] for _ in range(n)]


def test_sim_reset_key_is_a_zero_action_feature(clock):
    teleop = make_teleop(clock)
    assert SIM_RESET_KEY in teleop.action_features
    assert step(teleop, clock, XRFrame())[SIM_RESET_KEY] == 0.0


def test_b_changes_nothing(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation())
    frame = XRFrame(tracking=True, buttons={"b": True})
    assert resets(teleop, clock, frame, 200) == [0.0] * 200
    press(teleop, clock, "a")
    assert teleop.engaged
    assert resets(teleop, clock, frame, 200) == [0.0] * 200
    assert teleop.engaged


def test_targets_resync_from_feedback_for_a_second_after_the_reset(clock):
    teleop = make_teleop(clock)
    key = BODY_KEYS[0]
    teleop.send_feedback(observation())
    teleop.on_episode_end()
    assert step(teleop, clock, XRFrame(tracking=True))[SIM_RESET_KEY] == 1.0

    teleop.send_feedback(observation(**{key: 0.7}))
    assert step(teleop, clock)[key] == pytest.approx(0.7)
    step(teleop, clock, n=40)
    teleop.send_feedback(observation(**{key: 0.4}))
    assert step(teleop, clock)[key] == pytest.approx(0.4)

    step(teleop, clock, n=60)
    teleop.send_feedback(observation(**{key: -0.3}))
    assert step(teleop, clock)[key] == pytest.approx(0.4)


def test_no_resync_without_a_reset(clock):
    teleop = make_teleop(clock)
    key = BODY_KEYS[0]
    teleop.send_feedback(observation())
    teleop.send_feedback(observation(**{key: 0.7}))
    assert step(teleop, clock)[key] == pytest.approx(0.1)


def test_ready_to_record_follows_engagement(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation())
    assert teleop.ready_to_record() is False
    press(teleop, clock, "a")
    assert teleop.ready_to_record() is True
    press(teleop, clock, "a")
    assert teleop.ready_to_record() is False


def test_reset_sim_on_episode_end_defaults_on():
    assert UnitreeG1XRTeleopConfig().reset_sim_on_episode_end is True


def test_episode_end_disengages_and_resets_the_sim_once(clock):
    teleop = make_teleop(clock)
    teleop.send_feedback(observation())
    press(teleop, clock, "a")
    assert teleop.engaged
    teleop.on_episode_end()
    assert not teleop.engaged
    assert teleop.ready_to_record() is False
    values = [step(teleop, clock, XRFrame(tracking=True))[SIM_RESET_KEY] for _ in range(150)]
    assert values[0] == 1.0
    assert values.count(1.0) == 1


def test_episode_end_logs_one_info_line(clock, caplog):
    teleop = make_teleop(clock)
    with caplog.at_level("INFO"):
        teleop.on_episode_end()
    assert [r.getMessage() for r in caplog.records] == ["Episode ended: disengaged, resetting the sim"]


def test_episode_end_starts_the_resync_window(clock):
    teleop = make_teleop(clock)
    key = BODY_KEYS[0]
    teleop.send_feedback(observation())
    teleop.on_episode_end()
    assert step(teleop, clock, XRFrame())[SIM_RESET_KEY] == 1.0
    teleop.send_feedback(observation(**{key: 0.7}))
    assert step(teleop, clock)[key] == pytest.approx(0.7)
    step(teleop, clock, n=60)
    teleop.send_feedback(observation(**{key: -0.3}))
    assert step(teleop, clock)[key] == pytest.approx(0.7)


def test_episode_end_cancels_the_grip_hold(clock):
    teleop = make_teleop(clock)
    assert toggles(teleop, clock, grips_frame(), 30) == [0.0] * 30
    teleop.on_episode_end()
    assert teleop._grips_held_s == 0.0
    assert step(teleop, clock, grips_frame())[LOCOMOTION_TOGGLE_KEY] == 0.0
    assert toggles(teleop, clock, grips_frame(), 30) == [0.0] * 30


def test_episode_end_does_nothing_when_the_option_is_off(clock):
    teleop = make_teleop(clock, reset_sim_on_episode_end=False)
    teleop.send_feedback(observation())
    press(teleop, clock, "a")
    teleop.on_episode_end()
    assert teleop.engaged
    assert [step(teleop, clock, XRFrame(tracking=True))[SIM_RESET_KEY] for _ in range(100)] == [0.0] * 100
