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

"""Original-event capture for one Pi session branch with incremental recovery.

Implements the ``source-02-pi-history`` challenge: each visible event on the
selected branch becomes an independent ``CAPTURED`` Source identified by
``(session_id, event_id)``. ``PiSessionConnector`` replays the selected branch
and only submits events that were not already imported, persisting a checkpoint
that detects history rewrites.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"


class PiSessionInput(BaseModel):
    """Offline Pi v3 session archive bytes addressed by an optional leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One resolved visible event on the selected branch."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    """Captured evidence for a single Pi session event."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


def _pi_event_source_id(session_id: str, event_id: str) -> str:
    return f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"


def _canonical_record_hash(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _format_tool_call(block: dict[str, Any]) -> str:
    arguments = json.dumps(
        block.get("arguments", {}), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return f"[tool_call {block.get('name', '')} {block.get('id', '')}] {arguments}"


def _format_tool_result(message: dict[str, Any]) -> str:
    text_parts = [
        block.get("text", "")
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "\n".join(part for part in text_parts if part)
    return f"[tool_result {message.get('toolName', '')} {message.get('toolCallId', '')}] {text}"


def _format_bash_execution(message: dict[str, Any]) -> str:
    command = message.get("command", "")
    output = message.get("output", "")
    return f"$ {command}\n{output}"


def _visible_body(node: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(role, body)`` for a node with visible text, else ``None``."""

    node_type = node.get("type")
    if node_type == "message":
        message = node.get("message", {})
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role == "toolResult":
            return (role, _format_tool_result(message))
        if role == "bashExecution":
            return (role, _format_bash_execution(message))
        if isinstance(content, str):
            return (role, content)
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "toolCall":
                    parts.append(_format_tool_call(block))
            return (role, "\n".join(parts)) if parts else None
        return None
    if node_type in ("compaction", "branch_summary"):
        summary = node.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return (node_type, summary)
    if node_type == "custom_message":
        content = node.get("content")
        if not isinstance(content, list):
            return None
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(part for part in parts if part)
        return ("custom", text) if text else None
    # Control nodes without a visible body keep graph connectivity only.
    return None


def _parse_nodes(content: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("pi session content must be valid UTF-8") from error
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("pi session archive must contain a header record")
    nodes: list[dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("pi session archive must be valid JSONL") from error
        if not isinstance(record, dict):
            raise ValueError("pi session records must be JSON objects")
        nodes.append(record)
    header = nodes[0]
    if header.get("type") != "session" or header.get("version") != 3 or not isinstance(header.get("id"), str) or not header["id"]:
        raise ValueError("pi session header must be a v3 session record with a non-empty id")
    seen: set[str] = set()
    for node in nodes[1:]:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("pi session record id must be a non-empty string")
        if node_id in seen:
            raise ValueError("pi session record ids must be unique")
        parent_id = node.get("parentId")
        if parent_id is not None and parent_id not in seen:
            raise ValueError("pi session parent must reference an earlier record")
        seen.add(node_id)
    return header, nodes[1:]


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse the selected branch into visible ``PiEventInput`` events (root→leaf)."""

    header, records = _parse_nodes(value.content)
    session_id = header["id"]
    by_id = {record["id"]: record for record in records}
    leaf_id = value.leaf_id
    if leaf_id is None:
        leaf_id = records[-1]["id"]
    elif leaf_id not in by_id:
        raise ValueError("pi session leaf id must reference a known record")

    chain: list[str] = []
    cursor: str | None = leaf_id
    guard = 0
    while cursor is not None:
        if guard > len(records) + 1:
            raise ValueError("pi session graph must be acyclic")
        guard += 1
        chain.append(cursor)
        cursor = by_id[cursor].get("parentId")
    chain.reverse()

    events: list[PiEventInput] = []
    for event_id in chain:
        node = by_id[event_id]
        visible = _visible_body(node)
        if visible is None:
            continue
        role, body = visible
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=node.get("parentId"),
                event_type=node.get("type", ""),
                role=role,
                content=f"{role}: {body}",
                record=node,
            )
        )
    return tuple(events)


class PiEventSourceAdapter:
    """Resolve and read the Pi session event Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = _pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose a captured Pi event as canonical text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Incrementally import the selected Pi branch into one Scope."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = self._path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=raw, leaf_id=self._leaf_id))
        previous = session.checkpoint
        imported: dict[str, str] = {}
        if isinstance(previous, dict):
            imported = dict(previous.get("imported", {}))

        for event in events:
            event_id = event.event_id
            record_hash = _canonical_record_hash(event.record)
            if event_id in imported:
                if imported[event_id] != record_hash:
                    raise ValueError(f"imported event {event_id!r} was mutated after ingestion")
                continue
            await session.submit(event_id, PI_EVENT_SOURCE_NAME, event)
            imported[event_id] = record_hash

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"imported": imported},
        )


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
