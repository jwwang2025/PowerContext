# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-call embedding coalescing and a bounded vector cache.

Several scopes frequently embed the same text at the same time. A wrapper in
front of one model lets those callers share the work that is already running
and keeps the successful vectors for later calls, while every caller still
receives its own vectors in its own order and its own usage.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_COALESCE_MS = 5.0
DEFAULT_CACHE_SIZE = 1024


@dataclass(slots=True)
class _Batch:
    """One underlying request and the usage it reported."""

    texts: tuple[str, ...]
    usage: InferenceUsage = field(default_factory=lambda: InferenceUsage(requests=0))


@dataclass(slots=True)
class _Slot:
    """One text that is waiting for, or running inside, one batch."""

    future: asyncio.Future[EmbeddingVector]
    batch: _Batch | None = None


class CoalescingEmbeddingModel:
    """Share concurrent embedding work and cache the vectors that succeeded.

    The wrapper never changes the caller's view of the model: it exposes the
    underlying profile and returns one ordered vector per input text. Only
    the number of underlying requests changes.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = DEFAULT_COALESCE_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must not be negative")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_ms = coalesce_ms
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._slots: dict[str, _Slot] = {}
        self._pending: list[str] = []
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def profile(self):
        """Expose the underlying deployment-fixed profile."""

        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return one ordered vector per input text, sharing the work."""

        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        resolved: dict[str, EmbeddingVector] = {}
        slots: dict[str, _Slot] = {}
        for text in dict.fromkeys(texts):
            cached = self._cached(text)
            if cached is not None:
                resolved[text] = cached
                continue
            slot = self._slots.get(text)
            if slot is None:
                slot = _Slot(future=asyncio.get_running_loop().create_future())
                self._slots[text] = slot
                self._pending.append(text)
            slots[text] = slot
        if self._coalesce_ms > 0:
            # Give the other callers of this loop a chance to join the window
            # before any text leaves for the underlying model.
            await asyncio.sleep(self._coalesce_ms / 1000)
        self._dispatch()
        for text, slot in slots.items():
            # A cancelled caller must not cancel the shared request.
            resolved[text] = await asyncio.shield(slot.future)
        return EmbeddingResult(
            vectors=tuple(resolved[text] for text in texts),
            usage=self._usage(slots.values()),
        )

    def _cached(self, text: str) -> EmbeddingVector | None:
        """Return a cached vector and refresh its recency."""

        if text not in self._cache:
            return None
        self._cache.move_to_end(text)
        return self._cache[text]

    def _store(self, text: str, vector: EmbeddingVector) -> None:
        """Keep one successful vector within the bounded cache."""

        if self._cache_size <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _dispatch(self) -> None:
        """Send every waiting text to the underlying model in bounded batches."""

        if not self._pending:
            return
        waiting = tuple(self._pending)
        self._pending.clear()
        for start in range(0, len(waiting), self._max_batch_size):
            texts = waiting[start : start + self._max_batch_size]
            batch = _Batch(texts=texts)
            for text in texts:
                slot = self._slots.get(text)
                if slot is not None:
                    slot.batch = batch
            task = asyncio.create_task(self._run(batch), name=f"coalesced-embedding-{len(texts)}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run(self, batch: _Batch) -> None:
        """Run one underlying request and publish its vectors or its error."""

        try:
            result = await self._model.embed(batch.texts)
            if len(result.vectors) != len(batch.texts):
                raise ValueError(  # noqa: TRY003
                    f"embedding provider returned {len(result.vectors)} vectors for {len(batch.texts)} texts"
                )
            batch.usage = result.usage
        except BaseException as error:  # noqa: BLE001 - every waiter learns the failure
            for text in batch.texts:
                slot = self._slots.pop(text, None)
                if slot is None or slot.future.done():
                    continue
                slot.future.set_exception(error)
                slot.future.add_done_callback(_consume_exception)
            return
        for text, vector in zip(batch.texts, result.vectors, strict=True):
            slot = self._slots.pop(text, None)
            if slot is None:
                continue
            self._store(text, vector)
            if not slot.future.done():
                slot.future.set_result(vector)

    def _usage(self, waited) -> InferenceUsage:
        """Summarize the batches this call waited for, counting each once."""

        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        counted: set[int] = set()
        for slot in waited:
            batch = slot.batch
            if batch is None or id(batch) in counted:
                continue
            counted.add(id(batch))
            requests += batch.usage.requests
            if batch.usage.input_tokens is not None:
                input_tokens = (input_tokens or 0) + batch.usage.input_tokens
            if batch.usage.output_tokens is not None:
                output_tokens = (output_tokens or 0) + batch.usage.output_tokens
        return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)


def _consume_exception(future: asyncio.Future[EmbeddingVector]) -> None:
    """Retrieve a failure that no waiter observed, so it is not reported twice."""

    if not future.cancelled():
        future.exception()


__all__ = ["CoalescingEmbeddingModel"]
