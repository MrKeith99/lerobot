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

"""Hand out one action per control tick from action chunks predicted in a background thread.

Every chunk is anchored to the tick of the observation it was predicted from (its `origin`): at tick `t` the
action is `chunk[t - origin]`. Once the running chunk has been used for `execution_horizon` ticks, the current
observation goes to the worker and the running chunk keeps playing past its horizon until the new one is ready.
The new chunk then starts at the index of the current tick, so the actions that went stale while it was
computed are skipped. The control loop never waits for inference unless the running chunk runs out. One
observation per `execution_horizon` ticks is sent to the policy, as with blocking inference.

With `asynchronous=False` every request blocks and the new chunk starts at index 0 on the tick after its
observation was taken (the classic synchronous loop; the inference time stalls the control loop).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ChunkRecord:
    """One predicted chunk: observation tick, the tick it was first used, inference time and the actions."""

    origin: int
    ready_tick: int
    infer_s: float
    actions: np.ndarray = field(repr=False)


class ChunkScheduler:
    def __init__(
        self,
        predict: Callable[[Any], np.ndarray],
        execution_horizon: int,
        asynchronous: bool = True,
    ):
        """`predict(inputs)` returns a (T, action_dim) chunk; with `asynchronous` it runs in a worker thread
        (so anything thread-local, e.g. `torch.inference_mode`, must be entered inside it)."""
        if execution_horizon < 1:
            raise ValueError(f"execution_horizon must be >= 1, got {execution_horizon}")
        self.predict = predict
        self.execution_horizon = execution_horizon
        self.asynchronous = asynchronous
        self.chunks: list[ChunkRecord] = []
        self.blocked_ticks = 0  # ticks that waited for inference
        self._actions: np.ndarray | None = None
        self._origin = 0
        self._pending: Future | None = None
        self._pending_origin = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunk-inference")

    @property
    def chunk_id(self) -> int:
        """Index in `chunks` of the chunk the last action came from."""
        return len(self.chunks) - 1

    def _timed_predict(self, inputs: Any) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        actions = np.asarray(self.predict(inputs))
        return actions, time.perf_counter() - t0

    def _submit(self, tick: int, make_inputs: Callable[[], Any]) -> None:
        self._pending_origin = tick
        self._pending = self._executor.submit(self._timed_predict, make_inputs())

    def _adopt(self, tick: int) -> None:
        actions, infer_s = self._pending.result()  # re-raises an inference error here
        self._pending = None
        if actions.ndim != 2 or len(actions) == 0:
            raise ValueError(
                f"predict must return a non-empty (T, action_dim) chunk, got shape {actions.shape}"
            )
        self._actions, self._origin = actions, self._pending_origin
        self.chunks.append(ChunkRecord(self._origin, tick, infer_s, actions))

    def step(self, tick: int, make_inputs: Callable[[], Any]) -> np.ndarray:
        """The action for control tick `tick` (increasing by one per call). `make_inputs()` builds the policy
        inputs from this tick's observation; it is only called when a new chunk is requested."""
        if self._pending is not None and self._pending.done():
            self._adopt(tick)
        running = self._actions is not None
        if self._pending is None and (
            not running or tick - self._origin >= min(self.execution_horizon, len(self._actions))
        ):
            self._submit(tick, make_inputs)
        if self._pending is not None and (
            not running or not self.asynchronous or tick - self._origin >= len(self._actions)
        ):
            # Nothing to play meanwhile (first chunk, synchronous mode, running chunk used up): wait for it
            self.blocked_ticks += 1
            self._adopt(tick)
        index = tick - self._origin
        if index >= len(self._actions):
            raise RuntimeError(
                f"Inference took {index} ticks, longer than the {len(self._actions)}-action chunk"
            )
        return self._actions[index]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
