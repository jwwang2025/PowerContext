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

"""Source lifecycle node, dependency, and event persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENTS_TABLE,
    SOURCE_LIFECYCLE_NODES_TABLE,
    SOURCE_LIFECYCLE_PARENTS_TABLE,
)

SOURCE_LIFECYCLE_KIND_SOURCE = "source"
SOURCE_LIFECYCLE_KIND_MEMORY = "memory"
SOURCE_LIFECYCLE_KIND_TOPIC = "topic"

SOURCE_LIFECYCLE_POLICY_ALL = "all"
SOURCE_LIFECYCLE_POLICY_ANY = "any"


@dataclass(frozen=True, slots=True)
class StoredLifecycleNode:
    """One registered lifecycle node with its current availability."""

    scope_id: str
    node_id: str
    ordinal: int
    kind: str
    available: bool
    policy: str | None
    revision: int | None
    source_type: str | None
    source_id: str | None
    memory_artifact_id: str | None
    memory_revision: int | None
    entry_id: str | None
    entry_version_id: str | None
    topic_artifact_id: str | None
    topic_revision: int | None


@dataclass(frozen=True, slots=True)
class StoredLifecycleEvent:
    """One applied Source lifecycle event."""

    scope_id: str
    node_id: str
    sequence: int
    revision: int
    available: bool
    reason: str


class SourceLifecycleRepository:
    """Persist the evidence dependency graph and its Source lifecycle events."""

    async def node(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
    ) -> StoredLifecycleNode | None:
        row = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_NODES_TABLE).where(
                    SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
                )
            )
        ).mappings().one_or_none()
        return None if row is None else _decode_node(row)

    async def nodes(
        self,
        connection: AsyncConnection,
        scope_id: str,
        /,
    ) -> tuple[StoredLifecycleNode, ...]:
        rows = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_NODES_TABLE)
                .where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
                .order_by(SOURCE_LIFECYCLE_NODES_TABLE.c.ordinal)
            )
        ).mappings()
        return tuple(_decode_node(row) for row in rows)

    async def parents(
        self,
        connection: AsyncConnection,
        scope_id: str,
        /,
    ) -> dict[str, tuple[str, ...]]:
        rows = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_PARENTS_TABLE)
                .where(SOURCE_LIFECYCLE_PARENTS_TABLE.c.scope_id == scope_id)
                .order_by(
                    SOURCE_LIFECYCLE_PARENTS_TABLE.c.node_id,
                    SOURCE_LIFECYCLE_PARENTS_TABLE.c.ordinal,
                )
            )
        ).mappings()
        collected: dict[str, list[str]] = {}
        for row in rows:
            collected.setdefault(str(row["node_id"]), []).append(str(row["parent_id"]))
        return {node_id: tuple(parents) for node_id, parents in collected.items()}

    async def next_ordinal(self, connection: AsyncConnection, scope_id: str, /) -> int:
        current = await connection.scalar(
            select(func.max(SOURCE_LIFECYCLE_NODES_TABLE.c.ordinal)).where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
            )
        )
        return 0 if current is None else int(current) + 1

    async def insert_node(
        self,
        connection: AsyncConnection,
        node: StoredLifecycleNode,
        /,
        *,
        parents: Sequence[str] = (),
    ) -> None:
        await connection.execute(
            insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                scope_id=node.scope_id,
                node_id=node.node_id,
                ordinal=node.ordinal,
                kind=node.kind,
                available=node.available,
                policy=node.policy,
                revision=node.revision,
                source_type=node.source_type,
                source_id=node.source_id,
                memory_artifact_id=node.memory_artifact_id,
                memory_revision=node.memory_revision,
                entry_id=node.entry_id,
                entry_version_id=node.entry_version_id,
                topic_artifact_id=node.topic_artifact_id,
                topic_revision=node.topic_revision,
            )
        )
        if parents:
            await connection.execute(
                insert(SOURCE_LIFECYCLE_PARENTS_TABLE),
                [
                    {
                        "scope_id": node.scope_id,
                        "node_id": node.node_id,
                        "ordinal": ordinal,
                        "parent_id": parent,
                    }
                    for ordinal, parent in enumerate(parents)
                ],
            )

    async def set_availability(
        self,
        connection: AsyncConnection,
        scope_id: str,
        updates: Mapping[str, bool],
        /,
    ) -> None:
        for node_id, available in updates.items():
            await connection.execute(
                update(SOURCE_LIFECYCLE_NODES_TABLE)
                .where(
                    SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
                )
                .values(available=available)
            )

    async def event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        sequence: int,
        /,
    ) -> StoredLifecycleEvent | None:
        row = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_EVENTS_TABLE).where(
                    SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == scope_id,
                    SOURCE_LIFECYCLE_EVENTS_TABLE.c.node_id == node_id,
                    SOURCE_LIFECYCLE_EVENTS_TABLE.c.sequence == sequence,
                )
            )
        ).mappings().one_or_none()
        return None if row is None else _decode_event(row)

    async def max_sequence(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
    ) -> int:
        current = await connection.scalar(
            select(func.max(SOURCE_LIFECYCLE_EVENTS_TABLE.c.sequence)).where(
                SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_EVENTS_TABLE.c.node_id == node_id,
            )
        )
        return 0 if current is None else int(current)

    async def record_event(self, connection: AsyncConnection, event: StoredLifecycleEvent, /) -> None:
        await connection.execute(
            insert(SOURCE_LIFECYCLE_EVENTS_TABLE).values(
                scope_id=event.scope_id,
                node_id=event.node_id,
                sequence=event.sequence,
                revision=event.revision,
                available=event.available,
                reason=event.reason,
            )
        )

    async def withdrawn_topic_ids(
        self,
        connection: AsyncConnection,
        scope_id: str,
        /,
    ) -> frozenset[str]:
        rows = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_NODES_TABLE.c.topic_artifact_id).where(
                    SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.kind == SOURCE_LIFECYCLE_KIND_TOPIC,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.available.is_(False),
                )
            )
        ).scalars()
        return frozenset(str(row) for row in rows if row is not None)


def _decode_node(row: Mapping[Any, Any]) -> StoredLifecycleNode:
    return StoredLifecycleNode(
        scope_id=str(row["scope_id"]),
        node_id=str(row["node_id"]),
        ordinal=int(row["ordinal"]),
        kind=str(row["kind"]),
        available=bool(row["available"]),
        policy=None if row["policy"] is None else str(row["policy"]),
        revision=None if row["revision"] is None else int(row["revision"]),
        source_type=None if row["source_type"] is None else str(row["source_type"]),
        source_id=None if row["source_id"] is None else str(row["source_id"]),
        memory_artifact_id=(
            None if row["memory_artifact_id"] is None else str(row["memory_artifact_id"])
        ),
        memory_revision=None if row["memory_revision"] is None else int(row["memory_revision"]),
        entry_id=None if row["entry_id"] is None else str(row["entry_id"]),
        entry_version_id=None if row["entry_version_id"] is None else str(row["entry_version_id"]),
        topic_artifact_id=(
            None if row["topic_artifact_id"] is None else str(row["topic_artifact_id"])
        ),
        topic_revision=None if row["topic_revision"] is None else int(row["topic_revision"]),
    )


def _decode_event(row: Mapping[Any, Any]) -> StoredLifecycleEvent:
    return StoredLifecycleEvent(
        scope_id=str(row["scope_id"]),
        node_id=str(row["node_id"]),
        sequence=int(row["sequence"]),
        revision=int(row["revision"]),
        available=bool(row["available"]),
        reason=str(row["reason"]),
    )
