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

"""Row-level CSV capture with stable primary-key identity and text evidence.

Implements the ``source-01-table-evidence`` challenge: each CSV row becomes an
independent ``CAPTURED`` Source identified by ``(document_id, primary_key_value)``
so Memory citations can return to the exact original record after reordering or a
process restart.
"""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_ID_PREFIX = "csv:"


class CSVDocumentInput(BaseModel):
    """Offline CSV bytes addressed by a document and a primary key column."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One resolved CSV row keyed by its primary key value."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured evidence for a single CSV row."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the percent-encoded stable identity ``csv:{doc}:{key}``.

    Raises ``ValueError`` when ``document_id`` carries surrounding whitespace or
    the resulting identity exceeds the maximum source id length.
    """

    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without surrounding whitespace")
    source_id = f"{CSV_ROW_ID_PREFIX}{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("csv row source id must not exceed the maximum source id length")
    return source_id


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse an offline CSV document into one ``CSVRowInput`` per data row.

    Decoding is strict UTF-8 that tolerates a leading BOM. Parsing follows the
    standard library ``csv.reader`` semantics with ``strict=True``. The whole
    document is parsed and validated before any row is returned; any encoding,
    format, or data validation error raises ``ValueError`` without partial rows.
    """

    document_id = value.document_id
    primary_key = value.primary_key
    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without surrounding whitespace")
    if not primary_key:
        raise ValueError("primary_key must be a non-empty string")

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("csv document content must be valid UTF-8") from error

    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error as error:
        raise ValueError("csv document content must be valid CSV") from error

    if not rows:
        raise ValueError("csv document must contain a header row")
    header = rows[0]
    if not header:
        raise ValueError("csv document must contain a non-empty header")
    if any(field == "" or field.isspace() for field in header):
        raise ValueError("csv header fields must be non-empty and non-blank")
    if len(set(header)) != len(header):
        raise ValueError("csv header fields must be unique")

    try:
        key_index = header.index(primary_key)
    except ValueError as error:
        raise ValueError("primary_key must match exactly one header field") from error

    parsed: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for raw in rows[1:]:
        if not raw:
            # Completely empty physical record after the header is ignored.
            continue
        if len(raw) != len(header):
            raise ValueError("csv data row field count must match the header width")
        primary_key_value = raw[key_index]
        if primary_key_value == "" or primary_key_value.isspace():
            raise ValueError("csv primary key value must be a non-empty non-blank string")
        if primary_key_value in seen_keys:
            raise ValueError("csv primary key value must be unique within the document")
        seen_keys.add(primary_key_value)
        parsed.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=tuple(header),
                values=tuple(raw),
            )
        )
    return tuple(parsed)


class CSVRowSourceAdapter:
    """Resolve and read the row-level CSV Source."""

    input_class = CSVRowInput
    name = "csv-row"
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose a captured CSV row as canonical text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        mapping = {column: value for column, value in zip(source.columns, source.values, strict=True)}
        content = json.dumps(mapping, ensure_ascii=False, separators=(",", ":"))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=content,
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": mapping,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSVRowSourceAdapter(),
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
