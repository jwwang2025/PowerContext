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

"""Deterministic, prefix-reusing context packs for incremental transmission."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(Exception):
    """Raised on protocol, encoding, or chunking validation failures."""


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value, validate=True)


def _unsigned_items(items: tuple[bytes, ...]) -> list[str]:
    return [_b64encode(item) for item in items]


def _canonical_unsigned(items: tuple[bytes, ...]) -> bytes:
    payload = {"schema": PACK_SCHEMA, "items": _unsigned_items(items)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pack_digest(items: tuple[bytes, ...]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_unsigned(items)).hexdigest()


def _pack_byte_count(items: tuple[bytes, ...], digest: str) -> int:
    body = {"schema": PACK_SCHEMA, "items": _unsigned_items(items), "digest": digest}
    body_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # Full wire form is `{"byte_count":N,` + body_bytes[1:]; solve for N.
    prefix_len = len(body_bytes) + len(b'"byte_count":') + 1
    candidate = prefix_len
    for _ in range(8):
        nxt = prefix_len + len(str(candidate))
        if nxt == candidate:
            break
        candidate = nxt
    return candidate


def _chunk_content(content: str, chunk_bytes: int) -> tuple[bytes, ...]:
    if chunk_bytes < 1:
        raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
    if not content:
        return ()
    items: list[bytes] = []
    current = bytearray()
    for char in content:
        char_bytes = char.encode("utf-8")
        if len(char_bytes) > chunk_bytes:
            raise ContextPackError("a single character exceeds the chunk size")  # noqa: TRY003
        if current and len(current) + len(char_bytes) > chunk_bytes:
            items.append(bytes(current))
            current = bytearray()
        current.extend(char_bytes)
    if current:
        items.append(bytes(current))
    return tuple(items)


@dataclass(frozen=True)
class ContextPack:
    """A complete, self-describing chunked view of one prepared context body."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...] | list[bytes]) -> "ContextPack":
        items = tuple(items)
        for item in items:
            if not isinstance(item, (bytes, bytearray)):
                raise ContextPackError("context pack items must be bytes")  # noqa: TRY003
        digest = _pack_digest(items)
        return cls(items=items, digest=digest, byte_count=_pack_byte_count(items, digest))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> "ContextPack":
        return cls.build(_chunk_content(content, chunk_bytes))

    def to_bytes(self) -> bytes:
        body = {
            "schema": PACK_SCHEMA,
            "items": _unsigned_items(self.items),
            "digest": self.digest,
        }
        body_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return b'{"byte_count":' + str(self.byte_count).encode() + b',' + body_bytes[1:]

    @classmethod
    def from_bytes(cls, data: bytes) -> "ContextPack":
        try:
            payload = json.loads(data)
        except Exception as error:
            raise ContextPackError("context pack is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ContextPackError("context pack payload must be an object")  # noqa: TRY003
        if set(payload) != {"schema", "items", "digest", "byte_count"}:
            raise ContextPackError("context pack has unexpected or missing fields")  # noqa: TRY003
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("context pack schema mismatch")  # noqa: TRY003
        items_raw = payload["items"]
        if not isinstance(items_raw, list):
            raise ContextPackError("context pack items must be a list")  # noqa: TRY003
        items: list[bytes] = []
        for item in items_raw:
            try:
                decoded = _b64decode(item)
            except Exception as error:
                raise ContextPackError("context pack item is not valid base64") from error
            try:
                decoded.decode("utf-8")
            except Exception as error:
                raise ContextPackError("context pack item is not valid UTF-8") from error
            items.append(decoded)
        items = tuple(items)
        digest = payload["digest"]
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ContextPackError("context pack digest is invalid")  # noqa: TRY003
        if digest != _pack_digest(items):
            raise ContextPackError("context pack digest mismatch")  # noqa: TRY003
        byte_count = payload["byte_count"]
        if not isinstance(byte_count, int) or isinstance(byte_count, bool):
            raise ContextPackError("context pack byte_count is invalid")  # noqa: TRY003
        if byte_count != len(data):
            raise ContextPackError("context pack byte_count mismatch")  # noqa: TRY003
        return cls(items=items, digest=digest, byte_count=byte_count)


@dataclass(frozen=True)
class ContextPackDelta:
    """An incremental update referencing a known base pack by digest."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> "ContextPackDelta":
        prefix = 0
        for left, right in zip(base.items, target.items):
            if left == right:
                prefix += 1
            else:
                break
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=tuple(target.items[prefix:]),
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        body = {
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _unsigned_items(self.appended),
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ContextPackDelta":
        try:
            payload = json.loads(data)
        except Exception as error:
            raise ContextPackError("context pack delta is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ContextPackError("context pack delta payload must be an object")  # noqa: TRY003
        if set(payload) != {"schema", "base_digest", "target_digest", "prefix_items", "appended"}:
            raise ContextPackError("context pack delta has unexpected or missing fields")  # noqa: TRY003
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("context pack delta schema mismatch")  # noqa: TRY003
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not base_digest.startswith("sha256:"):
            raise ContextPackError("context pack delta base_digest is invalid")  # noqa: TRY003
        if not isinstance(target_digest, str) or not target_digest.startswith("sha256:"):
            raise ContextPackError("context pack delta target_digest is invalid")  # noqa: TRY003
        prefix = payload["prefix_items"]
        if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 0:
            raise ContextPackError("context pack delta prefix_items must be a non-negative integer")  # noqa: TRY003
        appended_raw = payload["appended"]
        if not isinstance(appended_raw, list):
            raise ContextPackError("context pack delta appended must be a list")  # noqa: TRY003
        appended: list[bytes] = []
        for item in appended_raw:
            try:
                decoded = _b64decode(item)
            except Exception as error:
                raise ContextPackError("context pack delta appended item is not valid base64") from error
            appended.append(decoded)
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix,
            appended=tuple(appended),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Reconstruct the target pack from a base pack and an incremental delta."""

    if not isinstance(base, ContextPack):
        raise ContextPackError("base must be a ContextPack")  # noqa: TRY003
    if base.digest != delta.base_digest:
        raise ContextPackError("context pack delta base_digest does not match the provided base")  # noqa: TRY003
    if delta.prefix_items < 0 or delta.prefix_items > len(base.items):
        raise ContextPackError("context pack delta prefix_items is out of range")  # noqa: TRY003
    reconstructed = base.items[: delta.prefix_items] + delta.appended
    target = ContextPack.build(reconstructed)
    if target.digest != delta.target_digest:
        raise ContextPackError("context pack delta target_digest mismatch")  # noqa: TRY003
    return target


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
