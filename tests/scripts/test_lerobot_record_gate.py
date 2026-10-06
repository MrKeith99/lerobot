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

import inspect
from unittest.mock import patch

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.scripts import lerobot_record  # noqa: E402
from lerobot.scripts.lerobot_record import RecordConfig, components_not_ready, wait_until_ready  # noqa: E402


class Ready:
    def __init__(self, ready_after: int | None = 0):
        self.ready_after = ready_after
        self.calls = 0

    def ready_to_record(self):
        self.calls += 1
        return self.ready_after is not None and self.calls > self.ready_after


class Plain:
    pass


def make_events():
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


def test_wait_for_operator_defaults_on():
    assert RecordConfig.__dataclass_fields__["wait_for_operator"].default is True


def test_components_without_the_method_are_ready():
    assert components_not_ready(Plain(), Plain()) == []
    assert components_not_ready(Plain(), [Plain(), Plain()]) == []


def test_components_not_ready_names_the_missing_ones():
    missing = components_not_ready(Ready(None), [Plain(), Ready(None), Ready(0)])
    assert len(missing) == 2
    assert "start locomotion" in missing[0]
    assert "press A" in missing[1]


def test_no_wait_when_ready():
    with patch("lerobot.scripts.lerobot_record.record_loop") as loop:
        wait_until_ready(Ready(0), [Ready(0)], make_events(), play_sounds=False)
    loop.assert_not_called()


def test_waits_until_readiness_flips_and_clears_flags():
    teleop = Ready(2)
    events = make_events()

    def fake_loop(**kwargs):
        events["exit_early"] = True
        events["rerecord_episode"] = True

    with patch("lerobot.scripts.lerobot_record.record_loop", side_effect=fake_loop) as loop:
        wait_until_ready(Plain(), teleop, events, play_sounds=False, fps=30)
    assert loop.call_count == 2
    assert loop.call_args.kwargs["control_time_s"] == 0.5
    assert loop.call_args.kwargs["teleop"] is teleop
    assert "dataset" not in loop.call_args.kwargs
    assert events["exit_early"] is False and events["rerecord_episode"] is False


def test_stop_during_wait_leaves_the_loop():
    events = make_events()

    def fake_loop(**kwargs):
        events["stop_recording"] = True

    with patch("lerobot.scripts.lerobot_record.record_loop", side_effect=fake_loop) as loop:
        wait_until_ready(Ready(None), Plain(), events, play_sounds=False)
    assert loop.call_count == 1
    assert events["stop_recording"] is True


class Notified:
    def __init__(self):
        self.ended = 0

    def on_episode_end(self):
        self.ended += 1


def test_notify_episode_end_calls_teleops_that_define_it():
    single, first, second = Notified(), Notified(), Notified()
    lerobot_record.notify_episode_end(single)
    lerobot_record.notify_episode_end([first, Plain(), second])
    lerobot_record.notify_episode_end(Plain())
    lerobot_record.notify_episode_end(None)
    assert (single.ended, first.ended, second.ended) == (1, 1, 1)


def test_record_notifies_right_after_the_episode_loop_before_the_reset_phase():
    source = inspect.getsource(lerobot_record.record)
    episode_loop = source.index("dataset=dataset,")
    notify = source.index("notify_episode_end(teleop)")
    reset_phase = source.index("Reset the environment")
    rerecord = source.index("clear_episode_buffer()")
    assert episode_loop < notify < reset_phase < rerecord
