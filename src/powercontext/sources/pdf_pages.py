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

import collections
import hashlib
import io
import math
import re
import warnings
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from powercontext.builtin.artifacts.memory.models import MemoryCitation, MemoryHit
from powercontext.builtin.artifacts.search import analyze_fts_query, analyze_text
from powercontext.builtin.runtime import SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

PDF_PAGE_ID_PREFIX = "pdf:"

# Retrieval pipeline constants. The runtime Memory search applies an admission
# floor over OR-matched FTS5 terms and truncates candidates, so a bare natural
# language question rarely surfaces the exact statement pages that answer it.
# The pipeline therefore pairs every query with the target document id (its
# tokens are the first line of every stored page entry), builds a wide
# candidate pool across per-term channels, and re-ranks that pool with
# document-local lexical features before returning exact citations.
_INFER_LIMIT = 48
_ENUM_LIMIT = 1000
_CHANNEL_LIMIT = 200
_ID_TOKEN_BONUS = 3.0
_BM25_K1 = 1.2
_BM25_B = 0.3
_W_TITLE = 1.0
_W_NUM_DENSITY = 1.0
_W_TOPIC_DENSITY_PENALTY = 0.5
_W_NUMBER_OVERLAP = 0.5
_W_RARE_TERM = 0.5

_ID_TOKEN_RE = re.compile(r"[a-z0-9]+")

# FinanceBench-style questionnaires prepend role-playing and fallback clauses
# that carry no evidence signal; stripping them sharpens both document
# inference and the per-term retrieval channels.
_BOILERPLATE_PATTERNS = (
    re.compile(r"\brespond to the question by assuming the perspective of [^.?!]*[.?!]?", re.I),
    re.compile(r"\bassume the perspective of [^.?!]*[.?!]?", re.I),
    re.compile(r"\bif [^.?!]*?\bnot\b[^.?!]*?,?\s*(then\s+)?(please\s+)?state (that|your answer) and explain why\.?\s*", re.I),
    re.compile(r"\bif [^.?!]*?\buseful metric\b[^.?!]*?(then\s+)?state that and explain why\.?\s*", re.I),
    re.compile(r"\bplease state your answer in terms of [^.?!]*[.?!]?", re.I),
    re.compile(r"\bplease (state|use|assume|respond)[^.?!]*[.?!]?", re.I),
    re.compile(r"\byour answer should [^.?!]*[.?!]?", re.I),
    re.compile(r"\bthe answer must be [^.?!]*[.?!]?", re.I),
)

# Questions mentioning any of these ask about a concrete figure and are best
# answered by dense statement tables; other questions are narrative topics
# where digit-heavy table pages are less likely to be the evidence.
_METRIC_PATTERN = re.compile(
    r"ebitda|gross margin|operating margin|net income|net profit|revenue|quick ratio|current ratio|"
    r"liquidity|balance sheet|income statement|cash flow statement|total assets|total liabilities|"
    r"leverage|net debt|working capital|inventory|acid.test|capex|capital expenditure|"
    r"depreciation and amortization|free cash flow|effective tax rate|tax rate|earnings per share|"
    r"dividend|buyback|share repurchase|operating income|cost of goods|cogs|gross profit|"
    r"operating expenses|return on|book value|days inventory|days payable|days receivable|"
    r"unadjusted|how much (money|cash|revenue|did|does)|profit margin|capital.intensive",
    re.I,
)

_INTENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"quick ratio|current ratio|liquidity|balance sheet|total assets|total liabilities|debt|leverage|net debt|working capital|inventory|acid.test|days (inventory|payable|receivable)", re.I), "balance_sheet"),
    (re.compile(r"ebitda|gross margin|operating margin|net income|net profit|revenue|income statement|gross profit|operating income|earnings per share|eps\b|cost of goods|operating expenses|effective tax rate|tax rate|book value|profit", re.I), "income_statement"),
    (re.compile(r"cash flow|capex|capital expenditure|depreciation and amortization|free cash flow|operating activities", re.I), "cash_flow"),
    (re.compile(r"capital.intensive|capex|capital expenditure", re.I), "balance_sheet"),
    (re.compile(r"capital.intensive|capex|capital expenditure", re.I), "cash_flow"),
)

# Statement titles are matched against whole page header lines only, so a body
# sentence that merely mentions "consolidated balance sheets" cannot fake the
# bonus. Different filers spell the titles differently; the set below covers
# the common spellings.
_STATEMENT_TITLE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "balance_sheet": (re.compile(r"^\s*consolidated\s+balance\s+sheets?\s*:?\s*$", re.I),),
    "income_statement": (
        re.compile(r"^\s*consolidated\s+statements?\s+of\s+(operations|income)\s*:?\s*$", re.I),
        re.compile(r"^\s*consolidated\s+statements?\s+of\s+income\s*$", re.I),
        re.compile(r"^\s*(table\s*\d+\s*:?\s*)?selected\s+(income\s+statement|financial)\s+data\s*$", re.I),
    ),
    "cash_flow": (re.compile(r"^\s*consolidated\s+statements?\s+of\s+cash\s+flows?\s*:?\s*$", re.I),),
}


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


def _strip_boilerplate(question: str) -> str:
    """Remove questionnaire boilerplate clauses that carry no evidence signal."""

    out = question
    for pattern in _BOILERPLATE_PATTERNS:
        out = pattern.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


def _content_terms(text: str) -> tuple[str, ...]:
    """Distinct lexical terms exactly as the runtime FTS query analyzer sees them."""

    return tuple(dict.fromkeys(analyze_fts_query(text).split()))


def _variants(term: str) -> tuple[str, ...]:
    """Morphological fallbacks for one term; FTS5 ``unicode61`` has no stemmer."""

    variants = {term}
    if len(term) > 6 and term.endswith("ies"):
        variants.update({term[:-3] + "y", term[:-3] + "ic"})
    if len(term) > 5 and term.endswith("ities"):
        variants.update({term[:-5] + "ity", term[:-5] + "e"})
    if len(term) > 5 and term.endswith("ality"):
        variants.add(term[:-5] + "al")
    if len(term) > 4 and term.endswith("ity"):
        variants.update({term[:-3] + "e", term[:-3]})
    if len(term) > 4 and term.endswith("ing"):
        variants.update({term[:-3], term[:-3] + "e"})
    if len(term) > 4 and term.endswith("es"):
        variants.add(term[:-2])
    if len(term) > 3 and term.endswith("s"):
        variants.add(term[:-1])
    if not term.endswith("s") and len(term) >= 3:
        variants.add(term + "s")
    if term.endswith("y") and len(term) >= 4:
        variants.add(term[:-1] + "ies")
    return tuple(variants)


def _question_type(question: str) -> str:
    return "metric" if _METRIC_PATTERN.search(question) else "topic"


def _title_intents(question: str) -> tuple[str, ...]:
    """Statement groups the question is plausibly about, in first-seen order."""

    return tuple(dict.fromkeys(name for pattern, name in _INTENT_PATTERNS if pattern.search(question)))


def _is_abbreviation(token: str, name: str) -> bool:
    """True when ``token`` reads as a short form of ``name`` (jnj→johnson, amex→americanexpress)."""

    return (
        len(name) >= 4
        and len(token) >= 3
        and token[0] == name[0]
        and set(token) <= set(name)
        and len(name) >= 2 * len(token)
    )


def _document_of(hit_text: str) -> str:
    """Entry texts open with the document id on their own line."""

    return hit_text.split("\n", 1)[0]


class _Page:
    """Per-entry lexical cache shared by every channel of one query."""

    __slots__ = ("entry_id", "memory_ref", "version_id", "doc", "term_set", "tf", "length", "lines")

    def __init__(self, hit: MemoryHit) -> None:
        self.entry_id = hit.entry_id
        self.memory_ref = hit.memory_ref
        self.version_id = hit.entry_version_id
        self.doc = _document_of(hit.text)
        tokens = analyze_text(hit.text).split()
        self.term_set = frozenset(tokens)
        self.tf = collections.Counter(tokens)
        self.length = len(tokens)
        self.lines = [line.strip().lower() for line in hit.text.splitlines() if line.strip()]

    @property
    def citation(self) -> MemoryCitation:
        return MemoryCitation(
            memory_ref=self.memory_ref,
            entry_id=self.entry_id,
            entry_version_id=self.version_id,
        )


async def _search(memory, query: str, limit: int) -> list[_Page]:
    """Run one lexical Memory search and cache the returned page entries."""

    if not query or limit <= 0:
        return []
    page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=limit))
    return [_Page(hit) for hit in page.hits]


async def _infer_document(memory, question: str, stripped: str, terms: tuple[str, ...]) -> str | None:
    """Infer the document a question targets using only public Memory searches.

    Three signals are combined: per-term coverage weighted by how many
    documents the term distinguishes, full-question search votes, and bonus
    credit when a question token matches (or abbreviates) a harvested
    document-id token.
    """

    coverage: dict[str, float] = collections.defaultdict(float)
    seen_docs: set[str] = set()

    for term in terms:
        pages = await _search(memory, term, _INFER_LIMIT)
        docs = {page.doc for page in pages}
        seen_docs.update(docs)
        if docs:
            weight = 1.0 / len(docs)
            for doc in docs:
                coverage[doc] += weight

    for rank, page in enumerate(await _search(memory, stripped, 32)):
        seen_docs.add(page.doc)
        coverage[page.doc] += 1.0 / (1.0 + rank)

    raw_tokens = set(_ID_TOKEN_RE.findall(question.lower()))
    for doc in seen_docs:
        id_tokens = set(_ID_TOKEN_RE.findall(doc.lower()))
        for token in raw_tokens:
            if token in id_tokens or any(_is_abbreviation(token, name) for name in id_tokens):
                coverage[doc] += _ID_TOKEN_BONUS

    if not coverage:
        return None
    return max(sorted(coverage), key=lambda doc: coverage[doc])


async def _document_stats(memory, doc: str) -> tuple[int, float]:
    """Enumerate one document's pages to get its page count and mean entry length."""

    pages = [page for page in await _search(memory, doc, _ENUM_LIMIT) if page.doc == doc]
    if not pages:
        return 1, 100.0
    return len(pages), sum(page.length for page in pages) / len(pages)


async def _candidate_pool(
    memory,
    doc: str,
    stripped: str,
    term_variants: list[tuple[str, tuple[str, ...]]],
) -> tuple[dict[str, _Page], dict[str, int]]:
    """Collect a wide same-document candidate pool over doc-paired channels.

    Each channel prefixes the document id to the query: those tokens are
    present on every page of the document, so target pages outrank other
    documents' pages inside the runtime's bm25-ordered candidate window while
    trivially clearing the admission floor. Document-local term frequencies
    for the idf-style features are measured from the same channels.
    """

    pool: dict[str, _Page] = {}

    def absorb(pages: list[_Page]) -> None:
        for page in pages:
            if page.doc == doc:
                pool[page.entry_id] = page

    absorb(await _search(memory, f"{doc} {stripped}", _CHANNEL_LIMIT))
    term_pages: dict[str, list[_Page]] = {}
    for term, _ in term_variants:
        pages = await _search(memory, f"{doc} {term}", _CHANNEL_LIMIT)
        doc_pages = [page for page in pages if page.doc == doc]
        term_pages[term] = doc_pages
        absorb(doc_pages)

    dfs: dict[str, int] = {}
    for term, variants_ in term_variants:
        wanted = set(variants_)
        dfs[term] = sum(1 for page in term_pages[term] if wanted.intersection(page.term_set))
    return pool, dfs


def _rank_pages(
    pool: dict[str, _Page],
    dfs: dict[str, int],
    term_variants: list[tuple[str, tuple[str, ...]]],
    question: str,
    n_pages: int,
    avgdl: float,
) -> list[str]:
    """Score the candidate pool with document-local lexical features."""

    qtype = _question_type(question)
    intents = _title_intents(question) if qtype == "metric" else ()
    question_digits = {term for term, _ in term_variants if term.isdigit()}
    scores: dict[str, float] = {}
    for entry_id, page in pool.items():
        bm25 = rare = 0.0
        for term, variants_ in term_variants:
            tf = sum(page.tf.get(variant, 0) for variant in variants_)
            df = dfs.get(term, 0)
            if not tf or not df:
                continue
            idf = math.log(1.0 + (n_pages - df + 0.5) / (df + 0.5))
            bm25 += idf * tf * (_BM25_K1 + 1.0) / (
                tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * page.length / avgdl)
            )
            rare = max(rare, idf / (1.0 + df))
        digits = sum(1 for token in page.term_set if token.isdigit())
        density = digits / max(len(page.term_set), 1)
        number_overlap = (
            len(question_digits.intersection(page.term_set)) / len(question_digits) if question_digits else 0.0
        )
        title_bonus = 0.0
        for intent in intents:
            for pattern in _STATEMENT_TITLE_PATTERNS.get(intent, ()):
                if any(pattern.match(line) for line in page.lines):
                    title_bonus = 1.0
                    break
        density_term = _W_NUM_DENSITY * (density if qtype == "metric" else -_W_TOPIC_DENSITY_PENALTY * density)
        scores[entry_id] = (
            bm25 / 10.0
            + _W_RARE_TERM * rare
            + density_term
            + _W_NUMBER_OVERLAP * number_overlap
            + _W_TITLE * title_bonus
        )
    return sorted(pool, key=lambda entry_id: (-scores[entry_id], entry_id))


async def search_pdf_pages(
    memory,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Search a scoped Memory for the pages that answer ``question``.

    Uses the public Memory FTS API and returns up to ``limit`` distinct exact
    ``MemoryCitation`` objects, one per matched physical page. The target
    document is inferred first, then every retrieval channel is scoped to that
    document by pairing queries with its id, and the merged candidate pool is
    re-ranked with document-local lexical features.
    """

    if limit <= 0:
        return ()
    stripped = _strip_boilerplate(question)
    terms = _content_terms(stripped)
    if not terms:
        return ()
    doc = await _infer_document(memory, question, stripped, terms)
    if doc is None:
        return ()
    n_pages, avgdl = await _document_stats(memory, doc)
    term_variants = [(term, _variants(term)) for term in terms]
    pool, dfs = await _candidate_pool(memory, doc, stripped, term_variants)
    ranked_ids = _rank_pages(pool, dfs, term_variants, question, n_pages, avgdl)
    return tuple(pool[entry_id].citation for entry_id in ranked_ids[:limit])


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "pdf_page_source_id",
    "parse_pdf_document",
    "search_pdf_pages",
]
