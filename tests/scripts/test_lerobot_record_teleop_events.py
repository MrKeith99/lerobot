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

from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.scripts import lerobot_record  # noqa: E402
from lerobot.teleoperators.teleoperator import Teleoperator  # noqa: E402
from lerobot.teleoperators.utils import TeleopEvents  # noqa: E402


class FakeTeleop(Teleoperator):
    name = "fake"

    def __init__(self, pending=()):
        self.pending = set(pending)
        self.config = MagicMock()

    action_features = {"a": float}
    feedback_features = {}
    is_connected = True
    is_calibrated = True

    def connect(self, calibrate=True):
        pass

    def calibrate(self):
        pass

    def configure(self):
        pass

    def send_feedback(self, feedback):
        pass

    def disconnect(self):
        pass

    def get_action(self):
        return {"a": 0.0}

    def get_teleop_events(self):
        pending, self.pending = self.pending, set()
        return {
            TeleopEvents.SUCCESS: TeleopEvents.SUCCESS in pending,
            TeleopEvents.RERECORD_EPISODE: TeleopEvents.RERECORD_EPISODE in pending,
        }


class NoEventsTeleop(FakeTeleop):
    get_teleop_events = None


def run_one_iteration(teleop, events):
    robot = MagicMock()
    robot.get_observation.return_value = {}
    identity = MagicMock(side_effect=lambda x: x)
    lerobot_record.record_loop(
        robot=robot,
        events=events,
        fps=1000,
        teleop_action_processor=identity,
        robot_action_processor=identity,
        robot_observation_processor=identity,
        teleop=teleop,
        control_time_s=0.0005,
    )
    return robot


def make_events():
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


def test_success_event_ends_the_episode():
    events = make_events()
    run_one_iteration(FakeTeleop({TeleopEvents.SUCCESS}), events)
    assert events["exit_early"] is True
    assert events["rerecord_episode"] is False


def test_rerecord_event_sets_both_flags():
    events = make_events()
    run_one_iteration(FakeTeleop({TeleopEvents.RERECORD_EPISODE}), events)
    assert events["exit_early"] is True
    assert events["rerecord_episode"] is True


def test_no_event_changes_nothing():
    events = make_events()
    run_one_iteration(FakeTeleop(), events)
    assert events == make_events()


def test_teleop_without_the_method_is_unaffected():
    events = make_events()
    run_one_iteration(NoEventsTeleop({TeleopEvents.SUCCESS}), events)
    assert events == make_events()


def test_list_teleops_are_unaffected():
    events = make_events()
    arm = FakeTeleop({TeleopEvents.SUCCESS})
    teleops = [arm, MagicMock()]
    lerobot_record.apply_teleop_events(teleops, events)
    assert events == make_events()
    assert arm.pending == {TeleopEvents.SUCCESS}
