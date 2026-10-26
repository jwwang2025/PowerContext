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

"""HTTP scoring-service adapter for Memory reranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class _RerankScore(BaseModel):
    """One score entry that must reference exactly one batch candidate rank."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    score: float


class _RerankResponse(BaseModel):
    """A scoring response that must contain exactly the ``scores`` collection."""

    model_config = ConfigDict(extra="forbid")

    scores: tuple[_RerankScore, ...]


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


@runtime_checkable
class _MemoryRerankerLike(Protocol):
    policy_id: str

    async def rerank(self, query: str, candidates: tuple[MemoryHit, ...], limit: int, /) -> MemoryRerankDecision: ...


class HTTPMemoryReranker:
    """Score candidates through an external HTTP reranking service.

    The service receives one request per batch of at most ``batch_size`` candidates,
    each tagged with its original global one-based ``rank``. Responses may arrive
    out of order but must cover exactly that batch's ranks once. The final selection
    is decided globally from the full candidate pool's scores: descending score,
    ties broken by ascending original rank.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if batch_size < 1:
            raise ValueError("rerank batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score and globally rerank ``candidates``, returning original ranks."""

        _validate_bounds(len(candidates), limit)

        ranked = list(enumerate(candidates, start=1))
        batches: list[list[tuple[int, MemoryHit]]] = []
        for start in range(0, len(ranked), self._batch_size):
            batches.append(ranked[start : start + self._batch_size])

        scored: list[tuple[int, float]] = []
        requests = 0
        for batch in batches:
            batch_ranks = [rank for rank, _ in batch]
            body = {
                "query": query,
                "candidates": [{"rank": rank, "text": hit.text} for rank, hit in batch],
            }
            response = await self._client.post(self._endpoint, json=body)
            response.raise_for_status()
            try:
                parsed = _RerankResponse.model_validate(response.json())
            except Exception as error:
                raise ValueError("rerank response is not valid JSON or has an invalid schema") from error
            response_ranks = [item.rank for item in parsed.scores]
            if len(response_ranks) != len(batch_ranks) or sorted(response_ranks) != sorted(batch_ranks):
                raise ValueError("rerank response must cover exactly this batch's ranks once each")  # noqa: TRY003
            for item in parsed.scores:
                if not math.isfinite(item.score):
                    raise ValueError("rerank score must be a finite number")  # noqa: TRY003
                scored.append((item.rank, item.score))
            requests += 1

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        selected_ranks = tuple(rank for rank, _ in scored[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected_ranks,
            usage=InferenceUsage(requests=requests),
            discarded_rank_count=len(candidates) - len(selected_ranks),
            used_fallback=False,
        )


__all__ = ["HTTPMemoryReranker"]
