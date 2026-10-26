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

"""Count admission for automatic Source processing.

A Family that owns a Source cursor also owns the count of journal identities
published after that cursor. The count is an admission signal, not a second
cursor: it is recomputed from durable state, so a repeated capture, a replayed
identity, and a restarted Runtime all observe the same number.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import SOURCE_JOURNAL_HEADS_TABLE

# One requester accepts the Family's own automatic window. It returns whether
# the Runtime now holds an accepted request for that Family.
SourceCountRequester = Callable[[str], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class SourceCountAdmission:
    """Start one Family window once enough new Source identities are pending.

    The pending count spans the whole Source journal, including the
    lineage-only identities that a Family excludes from its own evidence. A
    duplicate capture reuses its published position, so it never raises the
    count on its own.
    """

    database: AsyncDatabase
    binding_name: str
    threshold: int
    request_processing: SourceCountRequester

    def __post_init__(self) -> None:
        if not self.binding_name or self.binding_name != self.binding_name.strip():
            raise ValueError("binding_name must be nonempty and trimmed")  # noqa: TRY003
        if type(self.threshold) is not int or self.threshold < 1:
            raise ValueError("count threshold must be a positive integer")  # noqa: TRY003

    async def pending_count(self, scope_id: str, /) -> int:
        """Count the journal identities published after the Family's cursor."""

        async with self.database.transaction() as connection:
            return await self._pending_count(connection, scope_id)

    async def reached(self, scope_id: str, /) -> bool:
        """Report whether the pending count reached the configured threshold."""

        return await self.pending_count(scope_id) >= self.threshold

    async def request(self, scope_id: str, /) -> bool:
        """Accept one automatic window when the pending count reached the threshold."""

        async with self.database.transaction() as connection:
            # Freeze the journal head before reading the cursor, matching the
            # order that capture and the Topic Memory target use.
            await self._lock_head(connection, scope_id)
            if await self._pending_count(connection, scope_id) < self.threshold:
                return False
        return await self.request_processing(scope_id)

    async def _pending_count(self, connection: AsyncConnection, scope_id: str) -> int:
        head = await self._head(connection, scope_id)
        cursor = await SourceCursorRepository().load(connection, scope_id, self.binding_name)
        return max(0, head - (0 if cursor is None else cursor.cursor.sequence))

    @staticmethod
    async def _head(connection: AsyncConnection, scope_id: str) -> int:
        return int(
            await connection.scalar(
                select(SOURCE_JOURNAL_HEADS_TABLE.c.position).where(
                    SOURCE_JOURNAL_HEADS_TABLE.c.scope_id == scope_id
                )
            )
            or 0
        )

    @staticmethod
    async def _lock_head(connection: AsyncConnection, scope_id: str) -> None:
        await connection.execute(
            update(SOURCE_JOURNAL_HEADS_TABLE)
            .where(SOURCE_JOURNAL_HEADS_TABLE.c.scope_id == scope_id)
            .values(position=SOURCE_JOURNAL_HEADS_TABLE.c.position)
        )


__all__ = ["SourceCountAdmission", "SourceCountRequester"]
