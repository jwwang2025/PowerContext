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

"""Source lifecycle: withdraw and restore evidence-backed content.

A Scope owns one evidence dependency graph. Source nodes are the roots; every
Memory and Topic Memory node names the parents that support it and states how
many of them it needs (``all`` or ``any``). A Source lifecycle event changes
the availability of one root and the change propagates through the graph, so
content that lost its evidence leaves search and Context while content that
still has support stays usable. The registered body is never rewritten, so an
audit purpose can still read the exact version that was captured.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory.errors import (
    InvalidMemoryCitationError,
    MemoryEntryNotFoundError,
)
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.source_lifecycle import (
    SOURCE_LIFECYCLE_KIND_MEMORY,
    SOURCE_LIFECYCLE_KIND_SOURCE,
    SOURCE_LIFECYCLE_KIND_TOPIC,
    SOURCE_LIFECYCLE_POLICY_ALL,
    SOURCE_LIFECYCLE_POLICY_ANY,
    SourceLifecycleRepository,
    StoredLifecycleEvent,
    StoredLifecycleNode,
)
from powercontext.builtin.records import BaseValueNotFoundError
from powercontext.builtin.sources import validate_scope_id
from powercontext.errors import ArtifactNotFoundError, RevisionConflictError
from powercontext.sources import SourceRef

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from powercontext.builtin.runtime.application import BuiltinRuntime

MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH = 128

LifecyclePurpose = Literal["generation", "audit"]


class SourceLifecycleError(ValueError):
    """Raised when a lifecycle registration or event violates the contract."""


class SourceLifecycleEvent(BaseModel):
    """One Source lifecycle transition request."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    reason: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")  # noqa: TRY003
        return value


@dataclass(frozen=True, slots=True)
class SourceLifecycleServices:
    """Relational handles the lifecycle service needs beside the Runtime."""

    database: AsyncDatabase
    repository: SourceLifecycleRepository


@dataclass(frozen=True, slots=True)
class _MemoryTransition:
    """The Memory entries whose native state has to follow the graph."""

    deactivate: tuple[str, ...] = ()
    activate: tuple[str, ...] = ()


class ScopedSourceLifecycleService:
    """Register evidence bindings and propagate Source lifecycle changes."""

    def __init__(
        self,
        runtime: BuiltinRuntime,
        services: SourceLifecycleServices,
        scope_id: str,
        /,
        *,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self._runtime = runtime
        self._services = services
        self.scope_id = validate_scope_id(scope_id)
        self._lock = lock

    def _serialize(self):
        """Return the Scope's lifecycle lock, or a no-op when none was wired."""

        if self._lock is None:
            return nullcontext()
        return self._lock

    @asynccontextmanager
    async def _lifecycle_transaction(self):
        """Open one lifecycle transaction under the Scope's lifecycle lock."""

        async with self._serialize():
            async with self._services.database.transaction() as connection:
                yield connection

    async def register_source(
        self,
        node_id: str,
        source_ref: SourceRef,
        *,
        revision: int,
    ) -> None:
        """Bind one captured Source as the root of the evidence graph."""

        node = _require_node_id(node_id)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise SourceLifecycleError("lifecycle revision must be a positive integer")
        await self._require_readable_source(source_ref)
        async with self._lifecycle_transaction() as connection:
            existing = await self._services.repository.node(connection, self.scope_id, node)
            if existing is not None:
                if (
                    existing.kind == SOURCE_LIFECYCLE_KIND_SOURCE
                    and existing.revision == revision
                    and existing.source_type == source_ref.source_type
                    and existing.source_id == source_ref.source_id
                ):
                    return
                raise SourceLifecycleError(f"lifecycle node is already bound: {node}")
            ordinal = await self._services.repository.next_ordinal(connection, self.scope_id)
            await self._services.repository.insert_node(
                connection,
                StoredLifecycleNode(
                    scope_id=self.scope_id,
                    node_id=node,
                    ordinal=ordinal,
                    kind=SOURCE_LIFECYCLE_KIND_SOURCE,
                    available=True,
                    policy=None,
                    revision=revision,
                    source_type=source_ref.source_type,
                    source_id=source_ref.source_id,
                    memory_artifact_id=None,
                    memory_revision=None,
                    entry_id=None,
                    entry_version_id=None,
                    topic_artifact_id=None,
                    topic_revision=None,
                ),
            )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: Sequence[str],
        policy: str,
    ) -> None:
        """Bind one readable Memory entry to the parents that support it."""

        node = _require_node_id(node_id)
        parent_ids = _require_parents(parents)
        rule = _require_policy(policy)
        await self._require_readable_memory(citation)
        async with self._serialize():
            async with self._services.database.transaction() as connection:
                existing = await self._services.repository.node(connection, self.scope_id, node)
                if existing is not None:
                    if (
                        existing.kind == SOURCE_LIFECYCLE_KIND_MEMORY
                        and existing.entry_id == citation.entry_id
                        and existing.entry_version_id == citation.entry_version_id
                        and existing.memory_artifact_id == citation.memory_ref.artifact_id
                        and existing.policy == rule
                    ):
                        return
                    raise SourceLifecycleError(f"lifecycle node is already bound: {node}")
                available = await self._resolve_availability(connection, parent_ids, rule)

            # A derived node starts out with the availability its parents
            # already grant, so an entry registered behind withdrawn evidence
            # never becomes usable.
            if not available:
                await self._runtime.memory.for_scope(self.scope_id).set_entry_states(
                    deactivate=(citation.entry_id,),
                    reason="source lifecycle registration",
                )

            async with self._services.database.transaction() as connection:
                ordinal = await self._services.repository.next_ordinal(connection, self.scope_id)
                await self._services.repository.insert_node(
                    connection,
                    StoredLifecycleNode(
                        scope_id=self.scope_id,
                        node_id=node,
                        ordinal=ordinal,
                        kind=SOURCE_LIFECYCLE_KIND_MEMORY,
                        available=available,
                        policy=rule,
                        revision=None,
                        source_type=None,
                        source_id=None,
                        memory_artifact_id=citation.memory_ref.artifact_id,
                        memory_revision=citation.memory_ref.revision,
                        entry_id=citation.entry_id,
                        entry_version_id=citation.entry_version_id,
                        topic_artifact_id=None,
                        topic_revision=None,
                    ),
                    parents=parent_ids,
                )

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: Sequence[str],
        policy: str,
    ) -> None:
        """Bind one readable Topic Memory revision to the parents that support it."""

        node = _require_node_id(node_id)
        parent_ids = _require_parents(parents)
        rule = _require_policy(policy)
        if artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("topic lifecycle nodes require a topic-memory reference")
        await self._require_readable_topic(artifact_ref)
        async with self._lifecycle_transaction() as connection:
            existing = await self._services.repository.node(connection, self.scope_id, node)
            if existing is not None:
                if (
                    existing.kind == SOURCE_LIFECYCLE_KIND_TOPIC
                    and existing.topic_artifact_id == artifact_ref.artifact_id
                    and existing.topic_revision == artifact_ref.revision
                    and existing.policy == rule
                ):
                    return
                raise SourceLifecycleError(f"lifecycle node is already bound: {node}")
            available = await self._resolve_availability(connection, parent_ids, rule)
            ordinal = await self._services.repository.next_ordinal(connection, self.scope_id)
            await self._services.repository.insert_node(
                connection,
                StoredLifecycleNode(
                    scope_id=self.scope_id,
                    node_id=node,
                    ordinal=ordinal,
                    kind=SOURCE_LIFECYCLE_KIND_TOPIC,
                    available=available,
                    policy=rule,
                    revision=None,
                    source_type=None,
                    source_id=None,
                    memory_artifact_id=None,
                    memory_revision=None,
                    entry_id=None,
                    entry_version_id=None,
                    topic_artifact_id=artifact_ref.artifact_id,
                    topic_revision=artifact_ref.revision,
                ),
                parents=parent_ids,
            )

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one Source lifecycle event and return every available node."""

        async with self._serialize():
            async with self._services.database.transaction() as connection:
                node = await self._services.repository.node(connection, self.scope_id, event.node_id)
                if node is None:
                    raise SourceLifecycleError(f"unknown lifecycle node: {event.node_id}")
                if node.kind != SOURCE_LIFECYCLE_KIND_SOURCE:
                    raise SourceLifecycleError("lifecycle events only apply to Source nodes")
                if node.revision != event.revision:
                    raise SourceLifecycleError("lifecycle revision does not match the Source")
                replayed = await self._services.repository.event(
                    connection,
                    self.scope_id,
                    event.node_id,
                    event.sequence,
                )
                if replayed is not None:
                    if (
                        replayed.revision != event.revision
                        or replayed.available != event.available
                        or replayed.reason != event.reason
                    ):
                        raise SourceLifecycleError(
                            "lifecycle event sequence already carries other content"
                        )
                    return await self._available_nodes(connection)
                highest = await self._services.repository.max_sequence(
                    connection,
                    self.scope_id,
                    event.node_id,
                )
                if event.sequence <= highest:
                    raise SourceLifecycleError("lifecycle event sequence must increase")
                nodes = await self._services.repository.nodes(connection, self.scope_id)
                parents = await self._services.repository.parents(connection, self.scope_id)
                desired = _propagate(nodes, parents, event.node_id, event.available)
                transitions = {
                    item.node_id: desired[item.node_id]
                    for item in nodes
                    if item.available != desired[item.node_id]
                }
                transition = _memory_transition(nodes, transitions)

            # The native Memory state is written before the event is recorded.
            # A failure here leaves the event unapplied, so the same event can
            # be retried and the remaining propagation still completes.
            if transition.deactivate or transition.activate:
                await self._runtime.memory.for_scope(self.scope_id).set_entry_states(
                    deactivate=transition.deactivate,
                    activate=transition.activate,
                    reason=event.reason,
                )

            async with self._services.database.transaction() as connection:
                await self._services.repository.set_availability(connection, self.scope_id, transitions)
                await self._services.repository.record_event(
                    connection,
                    StoredLifecycleEvent(
                        scope_id=self.scope_id,
                        node_id=event.node_id,
                        sequence=event.sequence,
                        revision=event.revision,
                        available=event.available,
                        reason=event.reason,
                    ),
                )
        return sorted(node_id for node_id, available in desired.items() if available)

    async def resolve(
        self,
        node_id: str,
        *,
        purpose: LifecyclePurpose,
    ) -> ArtifactRef | MemoryCitation | SourceRef:
        """Return the original reference of one node for the requested purpose."""

        node = _require_node_id(node_id)
        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError(f"unknown lifecycle purpose: {purpose}")
        async with self._lifecycle_transaction() as connection:
            stored = await self._services.repository.node(connection, self.scope_id, node)
        if stored is None:
            raise SourceLifecycleError(f"unknown lifecycle node: {node}")
        if purpose == "generation" and not stored.available:
            raise SourceLifecycleError(f"lifecycle node is not available for generation: {node}")
        return _reference(stored)

    async def _resolve_availability(
        self,
        connection: AsyncConnection,
        parent_ids: Sequence[str],
        rule: str,
    ) -> bool:
        support: list[bool] = []
        for parent in parent_ids:
            stored = await self._services.repository.node(connection, self.scope_id, parent)
            if stored is None:
                raise SourceLifecycleError(f"unknown lifecycle parent: {parent}")
            support.append(stored.available)
        if rule == SOURCE_LIFECYCLE_POLICY_ALL:
            return all(support)
        return any(support)

    async def _available_nodes(self, connection: AsyncConnection) -> list[str]:
        nodes = await self._services.repository.nodes(connection, self.scope_id)
        return sorted(item.node_id for item in nodes if item.available)

    async def _require_readable_source(self, source_ref: SourceRef) -> None:
        if self._runtime._record_service is None:
            return
        try:
            await self._runtime.records.for_scope(self.scope_id).get_source(
                source_ref.source_type,
                source_ref.source_id,
            )
        except BaseValueNotFoundError:
            raise SourceLifecycleError(f"source is not readable: {source_ref.source_id}") from None

    async def _require_readable_memory(self, citation: MemoryCitation) -> None:
        from powercontext.builtin.runtime.models import GetMemoryEntryRequest

        try:
            await self._runtime.memory.for_scope(self.scope_id).get(
                GetMemoryEntryRequest(citation=citation)
            )
        except Exception as error:  # noqa: BLE001 - any read failure means unreadable
            raise SourceLifecycleError(f"memory citation is not readable: {citation.entry_id}") from error

    async def _require_readable_topic(self, artifact_ref: ArtifactRef) -> None:
        if self._runtime._record_service is None:
            return
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                artifact_ref.family,
                artifact_ref.artifact_id,
                artifact_ref.revision,
            )
        except BaseValueNotFoundError:
            raise SourceLifecycleError(
                f"topic memory revision is not readable: {artifact_ref.artifact_id}"
            ) from None


class SourceLifecycleApplication:
    """Select the scoped Source lifecycle application service."""

    def __init__(
        self,
        runtime: BuiltinRuntime,
        services: SourceLifecycleServices | None,
    ) -> None:
        self._runtime = runtime
        self._services = services
        self._locks: dict[str, asyncio.Lock] = {}

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleService:
        if self._services is None:
            raise SourceLifecycleError("the runtime does not provide Source lifecycle persistence")
        scope = validate_scope_id(scope_id)
        return ScopedSourceLifecycleService(
            self._runtime,
            self._services,
            scope,
            lock=self.lock(scope),
        )

    def lock(self, scope_id: str, /) -> asyncio.Lock:
        """Serialize the lifecycle operations of one Scope."""

        scope = validate_scope_id(scope_id)
        lock = self._locks.get(scope)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[scope] = lock
        return lock


def _require_node_id(node_id: object) -> str:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("lifecycle node id must be a non-empty string")
    if len(node_id) > MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH:
        raise SourceLifecycleError("lifecycle node id is too long")
    return node_id


def _require_parents(parents: object) -> tuple[str, ...]:
    if isinstance(parents, str) or not isinstance(parents, Sequence):
        raise SourceLifecycleError("lifecycle parents must be a sequence of node ids")
    if not parents:
        raise SourceLifecycleError("lifecycle nodes need at least one parent")
    resolved: list[str] = []
    for parent in parents:
        node = _require_node_id(parent)
        if node in resolved:
            raise SourceLifecycleError(f"lifecycle parent is repeated: {node}")
        resolved.append(node)
    return tuple(resolved)


def _require_policy(policy: object) -> str:
    if policy not in (SOURCE_LIFECYCLE_POLICY_ALL, SOURCE_LIFECYCLE_POLICY_ANY):
        raise SourceLifecycleError("lifecycle policy must be 'all' or 'any'")
    return str(policy)


def _propagate(
    nodes: Sequence[StoredLifecycleNode],
    parents: Mapping[str, tuple[str, ...]],
    source_id: str,
    source_available: bool,
) -> dict[str, bool]:
    """Compute the availability of every node after one Source change.

    The nodes are read in registration order, which is also a topological
    order of the graph: a derived node may only name parents that were
    registered before it.
    """

    availability = {item.node_id: item.available for item in nodes}
    availability[source_id] = source_available
    for item in nodes:
        if item.kind == SOURCE_LIFECYCLE_KIND_SOURCE:
            continue
        support = [availability.get(parent, False) for parent in parents.get(item.node_id, ())]
        if item.policy == SOURCE_LIFECYCLE_POLICY_ALL:
            availability[item.node_id] = bool(support) and all(support)
        else:
            availability[item.node_id] = any(support)
    return availability


def _memory_transition(
    nodes: Sequence[StoredLifecycleNode],
    transitions: Mapping[str, bool],
) -> _MemoryTransition:
    deactivate: list[str] = []
    activate: list[str] = []
    for item in nodes:
        if item.kind != SOURCE_LIFECYCLE_KIND_MEMORY:
            continue
        target = transitions.get(item.node_id)
        if target is None or item.entry_id is None:
            continue
        if target:
            activate.append(item.entry_id)
        else:
            deactivate.append(item.entry_id)
    return _MemoryTransition(deactivate=tuple(deactivate), activate=tuple(activate))


def _reference(stored: StoredLifecycleNode) -> ArtifactRef | MemoryCitation | SourceRef:
    if stored.kind == SOURCE_LIFECYCLE_KIND_SOURCE:
        return SourceRef(source_type=str(stored.source_type), source_id=str(stored.source_id))
    if stored.kind == SOURCE_LIFECYCLE_KIND_MEMORY:
        return MemoryCitation(
            memory_ref=ArtifactRef(
                family="memory",
                artifact_id=str(stored.memory_artifact_id),
                revision=int(stored.memory_revision or 0),
            ),
            entry_id=str(stored.entry_id),
            entry_version_id=str(stored.entry_version_id),
        )
    return ArtifactRef(
        family="topic-memory",
        artifact_id=str(stored.topic_artifact_id),
        revision=int(stored.topic_revision or 0),
    )


__all__ = [
    "ScopedSourceLifecycleService",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleServices",
]
