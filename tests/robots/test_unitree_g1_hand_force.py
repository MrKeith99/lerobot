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

"""Tests for the hardware-free `HandForceLimiter` and its CLI flags."""

import argparse

import pytest

from lerobot.robots.unitree_g1.hand_force import (
    HandForceLimiter,
    add_hand_force_args,
    decode_scs_load,
    hand_force_from_args,
)

N = "m"


def step(lim, goal, pos, load=0, temp=30, dt=0.02):
    goals, off, events = lim.limit({N: goal}, {N: pos}, {N: load}, {N: temp}, dt)
    return goals[N], off[N], events


def test_decode_scs_load():
    assert decode_scs_load(0) == 0
    assert decode_scs_load(300) == 300
    assert decode_scs_load(1024 | 300) == -300
    assert decode_scs_load(0x3FF) == 1023


def test_error_cap_both_directions_and_passthrough():
    lim = HandForceLimiter()
    assert step(lim, 600, 500)[0] == 530
    assert step(lim, 400, 500)[0] == 470
    assert step(lim, 510, 500)[0] == 510
    assert step(lim, 500, 500)[0] == 500


def test_short_overload_spike_does_not_latch():
    lim = HandForceLimiter()
    for _ in range(5):
        assert step(lim, 600, 500, load=800)[0] == 530
    assert step(lim, 600, 500, load=100)[0] == 530
    for _ in range(5):
        assert step(lim, 600, 500, load=800)[0] == 530


def test_overload_latches_after_overload_s_and_backs_off():
    lim = HandForceLimiter()
    results = [step(lim, 600, 500, load=800, dt=0.1) for _ in range(3)]
    assert [r[0] for r in results[:2]] == [530, 530]
    goal, _, events = results[2]
    assert goal == 485
    assert len(events) == 1 and "backing off" in events[0]


def test_overload_back_off_direction_when_pushing_down():
    lim = HandForceLimiter()
    goal = None
    for _ in range(3):
        goal, _, _ = step(lim, 400, 500, load=-800, dt=0.1)
    assert goal == 515


def test_hold_while_latched_then_unlatch_on_opening_command():
    lim = HandForceLimiter()
    for _ in range(3):
        step(lim, 600, 500, load=800, dt=0.1)
    for pos in (495, 485, 485):
        goal, _, events = step(lim, 600, pos, load=0)
        assert goal == 485
        assert events == []
    goal, _, events = step(lim, 480, 485, load=0)
    assert goal == 480
    assert len(events) == 1 and "released" in events[0]
    assert step(lim, 600, 480)[0] == 510


def test_unlatch_when_goal_near_present_on_release_side():
    lim = HandForceLimiter()
    for _ in range(3):
        step(lim, 600, 500, load=800, dt=0.1)
    goal, _, events = step(lim, 490, 495)
    assert goal == 490
    assert any("released" in e for e in events)


def test_overload_without_error_never_latches():
    lim = HandForceLimiter()
    for _ in range(20):
        assert step(lim, 500, 500, load=900, dt=0.1)[0] == 500


def test_temperature_torque_off_with_hysteresis():
    lim = HandForceLimiter()
    assert step(lim, 600, 500, temp=64)[1] is False
    goal, off, events = step(lim, 600, 500, temp=65)
    assert off is True and goal == 500 and len(events) == 1
    goal, off, events = step(lim, 600, 500, temp=66)
    assert off is True and events == []
    assert step(lim, 600, 500, temp=62)[1] is True
    goal, off, events = step(lim, 600, 500, temp=59)
    assert off is False and goal == 530 and len(events) == 1


def test_missing_feedback_passes_goal_through():
    lim = HandForceLimiter()
    goals, off, _ = lim.limit({N: 900}, {}, {}, {}, 0.02)
    assert goals == {N: 900} and off == {N: False}


def test_servos_are_independent():
    lim = HandForceLimiter()
    for _ in range(3):
        goals, _, _ = lim.limit(
            {"a": 600, "b": 600}, {"a": 500, "b": 500}, {"a": 800, "b": 0}, {"a": 30, "b": 30}, 0.1
        )
    assert goals == {"a": 485, "b": 530}


def test_flags_default_on_and_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hands", action="store_true")
    add_hand_force_args(parser)
    args = parser.parse_args([])
    assert args.hand_force_limit is True
    assert hand_force_from_args(args) == HandForceLimiter()

    args = parser.parse_args(
        ["--hand-max-err-ticks", "20", "--hand-load-limit", "400", "--hand-overload-s", "0.5"]
        + ["--hand-backoff-ticks", "10", "--hand-temp-limit", "60"]
    )
    assert hand_force_from_args(args) == HandForceLimiter(20, 400, 0.5, 10, 60)

    assert hand_force_from_args(parser.parse_args(["--no-hand-force-limit"])) is None
    assert hand_force_from_args(parser.parse_args(["--no-hands"])) is None


def test_run_g1_server_registers_the_flags():
    pytest.importorskip("unitree_sdk2py")
    from lerobot.robots.unitree_g1 import run_g1_server

    assert run_g1_server.add_hand_force_args is add_hand_force_args
