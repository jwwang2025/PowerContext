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

"""One native statement for the seekdb hybrid candidate query.

The full-text candidates and the vector candidates live in two projections,
so the frozen path issued one statement per channel and joined the two result
sets in the application. seekdb can answer both channels in a single
statement, and this module expresses that: one ``UNION ALL`` whose first
branch scans the full-text index and whose second branch scans the vector
index. The scope, Memory artifact, tag filter, and candidate limit stay
inside the database, and the application only fuses the two ordered channels.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import (
    EmbeddingProfile,
    MemoryCapabilities,
    MemoryChannelHit,
    MemoryProjection,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.errors import CapabilityNotSupportedError
from powercontext.builtin.persistence.memory_index import memory_channel_hits
from powercontext.builtin.persistence.oceanbase.memory_index import (
    OceanBaseMemoryFTSIndex,
    OceanBaseMemoryVectorIndex,
)
from powercontext.builtin.persistence.tags import memory_tag_parameters, memory_tag_sql
from pyobvector import VECTOR

_HYBRID_CANDIDATE_SQL = """
(SELECT
    f.memory_artifact_id AS memory_artifact_id,
    f.head_revision AS head_revision,
    f.entry_id AS entry_id,
    f.entry_version_id AS entry_version_id,
    fv.text AS text,
    NULL AS distance,
    'fts' AS channel
FROM pc_memory_entry_heads AS f
JOIN pc_memory_entry_versions AS fv
  ON fv.scope_id = f.scope_id
 AND fv.memory_artifact_id = f.memory_artifact_id
 AND fv.entry_version_id = f.entry_version_id
WHERE f.scope_id = :scope_id
  AND f.memory_artifact_id IN :memory_artifact_ids
  AND MATCH(f.searchable_text) AGAINST (:analyzed_query)
{fts_tag_filter}
ORDER BY MATCH(f.searchable_text) AGAINST (:analyzed_query) DESC,
         f.memory_artifact_id,
         f.entry_id,
         f.entry_version_id
LIMIT :candidate_limit)
UNION ALL
(SELECT
    m.memory_artifact_id AS memory_artifact_id,
    m.head_revision AS head_revision,
    m.entry_id AS entry_id,
    m.entry_version_id AS entry_version_id,
    mv.text AS text,
    l2_distance(m.embedding, :query_vector) AS distance,
    'vector' AS channel
FROM pc_memory_vector_entries AS m
JOIN pc_memory_entry_versions AS mv
  ON mv.scope_id = m.scope_id
 AND mv.memory_artifact_id = m.memory_artifact_id
 AND mv.entry_version_id = m.entry_version_id
WHERE m.scope_id = :scope_id
  AND m.memory_artifact_id IN :memory_artifact_ids
{vector_tag_filter}
ORDER BY l2_distance(m.embedding, :query_vector) APPROXIMATE
LIMIT :candidate_limit)
"""


class SeekDBHybridMemoryIndex:
    """Hybrid candidate retrieval over one native seekdb statement.

    The single-channel modes keep the statements of the frozen path, so only
    the hybrid candidate query changes. The two channels are returned in the
    order the database produced them, and the fusion above this index still
    decides the final ranking.
    """

    def __init__(
        self,
        fts: OceanBaseMemoryFTSIndex,
        vector: OceanBaseMemoryVectorIndex,
    ) -> None:
        self._fts = fts
        self._vector = vector
        self.capabilities = MemoryCapabilities(
            fts=True,
            vector=True,
            hybrid=True,
            tag_filter=True,
            embedding_profile=vector.profile,
        )
        self.tables = fts.tables + vector.tables

    async def initialize(self, connection: AsyncConnection, /) -> None:
        await self._fts.initialize(connection)
        await self._vector.initialize(connection)

    async def replace(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        await self._fts.replace(connection, scope_id, memory_ref, projections)
        await self._vector.replace(connection, scope_id, memory_ref, projections)

    async def delete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        entry_ids: tuple[str, ...],
        /,
    ) -> None:
        await self._fts.delete(connection, scope_id, memory_ref, entry_ids)
        await self._vector.delete(connection, scope_id, memory_ref, entry_ids)

    async def upsert(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        await self._fts.upsert(connection, scope_id, memory_ref, projections)
        await self._vector.upsert(connection, scope_id, memory_ref, projections)

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        if request.mode != "hybrid":
            if request.mode == "fts":
                return await self._fts.search(connection, scope_id, request)
            if request.mode == "vector":
                return await self._vector.search(connection, scope_id, request)
            return MemorySearchChannels()
        if request.embedding_profile != self._vector.profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        await self._vector.vector_complete(connection, scope_id, request.memories, self._vector.profile)
        rows = (
            await connection.execute(
                self._hybrid_sql(request.tag_filter),
                {
                    "scope_id": scope_id,
                    "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
                    "analyzed_query": request.analyzed_query,
                    "query_vector": request.query_vector,
                    "candidate_limit": request.candidate_limit,
                    **memory_tag_parameters(request.tag_filter),
                },
            )
        ).mappings()
        fts: list[MemoryChannelHit] = []
        vector: list[MemoryChannelHit] = []
        for row in rows:
            if str(row["channel"]) == "fts":
                fts.extend(memory_channel_hits((row,), request.memories))
            else:
                vector.extend(memory_channel_hits((row,), request.memories))
        return MemorySearchChannels(fts=tuple(fts), vector=tuple(vector))

    async def vector_complete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memories: tuple[ArtifactRef, ...],
        profile: EmbeddingProfile,
        /,
    ) -> bool:
        return await self._vector.vector_complete(connection, scope_id, memories, profile)

    async def hydrate(
        self,
        connection: AsyncConnection,
        scope_id: str,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> tuple[MemoryProjection, ...]:
        return await self._vector.hydrate(connection, scope_id, projections)

    def _hybrid_sql(self, tag_filter: Any) -> Any:
        """Render the single statement with the tag filter applied in-database."""

        fts_filter = "" if tag_filter is None else memory_tag_sql("f")
        vector_filter = "" if tag_filter is None else memory_tag_sql("m")
        statement = _HYBRID_CANDIDATE_SQL.replace("{fts_tag_filter}", fts_filter).replace(
            "{vector_tag_filter}", vector_filter
        )
        if tag_filter is not None:
            # A tagged candidate query filters inside the database, so the
            # vector branch searches exactly instead of approximately. This
            # keeps the tagged hybrid result identical to the frozen path.
            statement = statement.replace(" APPROXIMATE", "")
        binds: list[Any] = [
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("query_vector", type_=VECTOR(self._vector.profile.dimension)),
        ]
        if tag_filter is not None:
            binds.extend([bindparam("tag_keys", expanding=True), bindparam("tag_hashes", expanding=True)])
        return text(statement).bindparams(*binds)


__all__ = ["SeekDBHybridMemoryIndex"]
