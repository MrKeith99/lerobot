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

import threading

import numpy as np
import pytest

from lerobot.utils.chunk_scheduler import ChunkScheduler

CHUNK = 40


class GatedPolicy:
    """Predicts a chunk whose action i is (observation tick, i). With `gated`, each prediction waits until the
    test calls `finish()`, so the test decides on which tick inference completes."""

    def __init__(self, gated: bool, chunk: int = CHUNK):
        self.gated = gated
        self.chunk = chunk
        self.calls: list[int] = []
        self._release = threading.Semaphore(0)
        self.started = threading.Event()

    def __call__(self, obs_tick: int) -> np.ndarray:
        self.calls.append(obs_tick)
        self.started.set()
        if self.gated:
            assert self._release.acquire(timeout=5.0), "the test never finished this inference"
        return np.array([[obs_tick, i] for i in range(self.chunk)], dtype=float)

    def finish(self) -> None:
        self._release.release()


def run(
    scheduler: ChunkScheduler, policy: GatedPolicy, ticks: range, finish_at: dict[int, int] | None = None
):
    """Step the scheduler; `finish_at[tick] = n` lets the pending inference finish just before that tick."""
    out = []
    for tick in ticks:
        if finish_at and tick in finish_at:
            policy.finish()
            pending = scheduler._pending
            if pending is not None:
                pending.result(timeout=5.0)  # done before the scheduler looks at it
        out.append(tuple(scheduler.step(tick, lambda t=tick: t)))
    return out


def test_synchronous_matches_the_blocking_loop():
    policy = GatedPolicy(gated=False)
    scheduler = ChunkScheduler(policy, execution_horizon=16, asynchronous=False)
    actions = run(scheduler, policy, range(48))
    assert policy.calls == [0, 16, 32]
    assert actions == [(16 * (t // 16), t % 16) for t in range(48)]
    assert scheduler.blocked_ticks == 3
    scheduler.close()


def test_asynchronous_keeps_playing_the_old_chunk_and_skips_stale_actions():
    policy = GatedPolicy(gated=True)
    scheduler = ChunkScheduler(policy, execution_horizon=16)
    # The first chunk has nothing to play meanwhile: let it finish before tick 0
    policy.finish()
    actions = run(scheduler, policy, range(40), finish_at={19: 1, 35: 1})
    assert policy.calls == [0, 16, 32]
    # Ticks 16-18 play the first chunk past its horizon; inference done at tick 19 -> index 3 of chunk 16
    assert actions[15:20] == [(0, 15), (0, 16), (0, 17), (0, 18), (16, 3)]
    assert actions[31:36] == [(16, 15), (16, 16), (16, 17), (16, 18), (32, 3)]
    # Every action is the one predicted for its own tick
    assert all(origin + index == t for t, (origin, index) in enumerate(actions))
    assert scheduler.blocked_ticks == 1  # only the first chunk
    assert [(c.origin, c.ready_tick) for c in scheduler.chunks] == [(0, 0), (16, 19), (32, 35)]
    scheduler.close()


def test_asynchronous_waits_only_when_the_running_chunk_runs_out():
    policy = GatedPolicy(gated=False, chunk=20)
    scheduler = ChunkScheduler(policy, execution_horizon=16)
    policy.gated = True
    policy.finish()  # first chunk
    scheduler.step(0, lambda: 0)
    for tick in range(1, 20):
        scheduler.step(tick, lambda t=tick: t)
    assert scheduler.blocked_ticks == 1
    # Tick 20: chunk 0 (20 actions) is used up and chunk 16 is not done: the step waits for it
    finisher = threading.Timer(0.05, policy.finish)
    finisher.start()
    assert tuple(scheduler.step(20, lambda: 20)) == (16, 4)
    assert scheduler.blocked_ticks == 2
    scheduler.close()


def test_execution_horizon_longer_than_the_chunk_requests_when_it_runs_out():
    policy = GatedPolicy(gated=False, chunk=10)
    scheduler = ChunkScheduler(policy, execution_horizon=16, asynchronous=False)
    actions = run(scheduler, policy, range(25))
    assert policy.calls == [0, 10, 20]
    assert actions[9:11] == [(0, 9), (10, 0)]
    scheduler.close()


def test_inference_error_reaches_the_control_loop():
    def broken(_):
        raise RuntimeError("CUDA out of memory")

    scheduler = ChunkScheduler(broken, execution_horizon=16)
    with pytest.raises(RuntimeError, match="out of memory"):
        scheduler.step(0, lambda: 0)
    scheduler.close()


def test_inputs_are_only_built_when_a_chunk_is_requested():
    policy = GatedPolicy(gated=False)
    scheduler = ChunkScheduler(policy, execution_horizon=16, asynchronous=False)
    built = []
    for tick in range(33):
        scheduler.step(tick, lambda t=tick: built.append(t) or t)
    assert built == [0, 16, 32]
    scheduler.close()


@pytest.mark.parametrize("bad", [0, -1])
def test_execution_horizon_must_be_positive(bad):
    with pytest.raises(ValueError, match="execution_horizon"):
        ChunkScheduler(lambda x: x, execution_horizon=bad)


def test_bad_chunk_shape_is_rejected():
    scheduler = ChunkScheduler(lambda _: np.zeros(5), execution_horizon=4)
    with pytest.raises(ValueError, match="action_dim"):
        scheduler.step(0, lambda: 0)
    scheduler.close()
