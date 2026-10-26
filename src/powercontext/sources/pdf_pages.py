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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full financial-statement PDF parsing with per-page evidence retrieval.

Implements the ``source-03-pdf-retrieval`` challenge: every physical page of a
complete source PDF is parsed independently, addressed by
``(document_id, document_sha256, page_number)`` so a later natural-language
query can return exact, re-checkable page citations from a scoped Memory.
"""

from __future__ import annotations

import hashlib
import io
import re
import warnings
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime import SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

PDF_PAGE_ID_PREFIX = "pdf:"

# Tokens shorter than this or in this set are dropped when building the
# expansion query used to fill recall slots below the requested limit.
_KEYWORD_MIN_LENGTH = 3
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by", "for",
        "with", "as", "is", "are", "was", "were", "be", "been", "being", "that",
        "this", "these", "those", "it", "its", "from", "into", "than", "then",
        "they", "their", "them", "we", "our", "you", "your", "he", "she", "his",
        "her", "what", "which", "who", "whom", "where", "when", "why", "how",
        "does", "did", "do", "does", "can", "could", "should", "would", "may",
        "might", "has", "have", "had", "will", "shall", "not", "no", "yes",
        "about", "between", "within", "during", "before", "after", "over",
        "under", "above", "below", "any", "all", "each", "every", "both",
        "such", "same", "other", "another", "there", "here", "out", "up", "down",
        "most", "more", "less", "much", "many", "few", "also", "than", "per",
        "via", "due", "based", "related", "including", "include", "report",
        "statement", "company", "companies", "inc", "corp", "annual", "year",
        "years", "fy", "q1", "q2", "q3", "q4",
    }
)


class PDFDocumentInput(BaseModel):
    """Offline PDF bytes addressed by a document id."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One resolved physical PDF page keyed by its document summary and number."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured evidence for a single physical PDF page."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the percent-encoded stable identity ``pdf:{doc}:{sha}:{page}``.

    Raises ``ValueError`` when ``document_id`` carries surrounding whitespace or
    the resulting identity exceeds the maximum source id length.
    """

    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without surrounding whitespace")
    source_id = (
        f"{PDF_PAGE_ID_PREFIX}{quote(document_id, safe='')}"
        f":{quote(document_sha256, safe='')}:{page_number}"
    )
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("pdf page source id must not exceed the maximum source id length")
    return source_id


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse a complete source PDF into one ``PDFPageInput`` per physical page.

    Decoding uses the shared locked ``pypdf``. All physical pages are returned in
    order with ``page_number`` starting at ``1``, including blank pages. The
    document summary is the lowercase hex SHA-256 of the entire input bytes.
    Empty files, non-PDF bytes, corrupted or otherwise un-parseable documents
    raise ``ValueError`` without yielding partial pages.
    """

    document_id = value.document_id
    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without surrounding whitespace")
    content = value.content
    if not content:
        raise ValueError("pdf document content must not be empty")
    document_sha256 = hashlib.sha256(content).hexdigest()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = PdfReader(io.BytesIO(content))
    except PdfReadError as error:
        raise ValueError("pdf document is not a valid or complete PDF") from error
    except Exception as error:  # noqa: BLE001 - pypdf can raise assorted errors on corruption
        raise ValueError("pdf document is not a valid or complete PDF") from error

    try:
        pages = reader.pages
        if not pages:
            raise ValueError("pdf document contains no pages")
        parsed: list[PDFPageInput] = []
        for index, page in enumerate(pages):
            try:
                extracted = page.extract_text()
            except Exception as error:  # noqa: BLE001 - any per-page failure is fatal
                raise ValueError(f"failed to extract pdf page {index + 1}") from error
            text = "" if extracted is None else extracted
            parsed.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    page_number=index + 1,
                    text=text,
                )
            )
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - unexpected structural failure
        raise ValueError("pdf document is not a valid or complete PDF") from error

    return tuple(parsed)


class PDFPageSourceAdapter:
    """Resolve and read the per-page PDF Source."""

    input_class = PDFPageInput
    name = "pdf-page"
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose a captured PDF page as canonical text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    projections=(PDFPageTextEvidenceProjection(),),
)


def _keyword_query(question: str, limit_tokens: int = 8) -> str:
    """Build a compact keyword query from a natural-language question."""

    tokens = re.findall(r"[A-Za-z0-9]+", question.lower())
    kept = [token for token in tokens if len(token) >= _KEYWORD_MIN_LENGTH and token not in _STOPWORDS]
    if not kept:
        return ""
    return " ".join(kept[:limit_tokens])


async def search_pdf_pages(
    memory,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Search a scoped Memory for the pages that answer ``question``.

    Uses the public Memory FTS API and returns up to ``limit`` distinct exact
    ``MemoryCitation`` objects, one per matched physical page. When the primary
    query returns fewer hits than requested, a keyword-only expansion is run to
    widen coverage without displacing the most relevant results.
    """

    seen: set[str] = set()
    citations: list[MemoryCitation] = []

    async def _collect(query: str, requested: int) -> None:
        if not query:
            return
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=requested))
        for hit in page.hits:
            if hit.entry_id in seen:
                continue
            seen.add(hit.entry_id)
            citations.append(
                MemoryCitation(
                    memory_ref=hit.memory_ref,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
            )

    await _collect(question, limit)
    if len(citations) < limit:
        await _collect(_keyword_query(question), limit - len(citations))

    return tuple(citations[:limit])


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "pdf_page_source_id",
    "parse_pdf_document",
    "search_pdf_pages",
]
