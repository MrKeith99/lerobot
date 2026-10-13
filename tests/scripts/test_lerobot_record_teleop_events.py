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

from lerobot.datasets.deleted_episodes import read_deleted_episodes  # noqa: E402
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
            TeleopEvents.DELETE_LAST_EPISODE: TeleopEvents.DELETE_LAST_EPISODE in pending,
        }


class NoEventsTeleop(FakeTeleop):
    get_teleop_events = None


def run_one_iteration(teleop, events, dataset=None, on_delete_last=None):
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
        dataset=dataset,
        on_delete_last=on_delete_last,
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


def test_delete_last_event_calls_the_handler_outside_an_episode():
    events, handler = make_events(), MagicMock()
    run_one_iteration(FakeTeleop({TeleopEvents.DELETE_LAST_EPISODE}), events, on_delete_last=handler)
    handler.assert_called_once_with()
    assert events == make_events(), "no flag is left behind and the loop is not ended"


def test_delete_last_event_is_ignored_while_recording():
    events, handler = make_events(), MagicMock()
    dataset = MagicMock(fps=1000)
    run_one_iteration(FakeTeleop({TeleopEvents.DELETE_LAST_EPISODE}), events, dataset, handler)
    handler.assert_not_called()


def test_record_status_reaches_only_teleops_with_the_hook():
    class WithStatus(FakeTeleop):
        def on_record_status(self, status):
            self.status = status

    teleop = WithStatus()
    lerobot_record.notify_record_status(teleop, state="idle")
    lerobot_record.notify_record_status([FakeTeleop(), teleop], state="reset")
    lerobot_record.notify_record_status(FakeTeleop(), state="idle")
    assert teleop.status == {"state": "reset"}


def test_record_session_tracks_the_deletable_episode(tmp_path):
    class WithStatus(FakeTeleop):
        def __init__(self):
            super().__init__()
            self.history = []

        def on_record_status(self, status):
            self.history.append(status)

    dataset = MagicMock(num_episodes=5, root=tmp_path)
    dataset.save_episode.side_effect = lambda: setattr(dataset, "num_episodes", dataset.num_episodes + 1)
    teleop = WithStatus()
    session = lerobot_record.RecordSession(dataset, teleop, target=3, episode_time_s=10, play_sounds=False)
    session.delete_last()  # nothing saved in this session
    assert read_deleted_episodes(tmp_path) == []
    session.save()
    assert (session.recorded, session.last_saved) == (1, 5)
    assert teleop.history[-1]["result"] == {"kind": "saved", "episode": 5}
    session.save()
    session.delete_last()
    assert (session.recorded, session.last_saved) == (1, None)
    assert read_deleted_episodes(tmp_path) == [6]
    assert teleop.history[-1]["deletable"] is None and teleop.history[-1]["kept"] == 1
    session.delete_last()
    assert read_deleted_episodes(tmp_path) == [6], "only the last episode is deletable, once"
    session.discard()
    dataset.clear_episode_buffer.assert_called_once()
    assert session.recorded == 1
    assert teleop.history[-1]["result"] == {"kind": "discarded", "episode": 7}
