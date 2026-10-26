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

"""Query-driven literal fragment selection over recalled Memory bodies.

The retrieval stage already decided which Memory entries are relevant. This
module decides which part of a recalled body is worth the caller's byte
budget. Every fragment is a verbatim slice of the original body, so the caller
can always point back at the exact characters that were selected.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.search import analyze_text_with_spans

# Sentence boundaries follow the same rule the FTS query analyzer uses.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

# A recalled body usually opens with its own heading. The retrieval stage
# already matched that heading, so the leading sentence is demoted and the
# budget goes to the body text that answers the query.
_LEADING_SENTENCE_DEMOTION = 0.4

# The evidence of one claim is usually a coherent passage, so a sentence that
# sits next to matching sentences is a better anchor than an isolated match.
_NEIGHBOUR_WEIGHT = 1.0

_STEM_SUFFIXES = ("ing", "edly", "ed", "es", "s")


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    """One recalled Memory body that may contribute literal fragments."""

    scope_id: str
    memory_ref: ArtifactRef
    entry_id: str
    entry_version_id: str
    text: str


@dataclass(frozen=True, slots=True)
class SelectedFragment:
    """One verbatim slice of a recalled Memory body."""

    scope_id: str
    memory_ref: ArtifactRef
    entry_id: str
    entry_version_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class _Sentence:
    """One candidate sentence with its score against the query."""

    document: int
    start: int
    end: int
    text: str
    score: float
    index: int = 0


def select_evidence_fragments(
    *,
    query: str,
    documents: Sequence[EvidenceDocument],
    max_bytes: int,
    measure: Callable[[Sequence[SelectedFragment]], int],
) -> tuple[SelectedFragment, ...]:
    """Select the literal fragments that answer ``query`` within a byte budget.

    ``measure`` reports the rendered size of a candidate fragment set, so the
    budget covers the whole output - body, citation, and envelope - instead of
    the selected text alone. A fragment is only kept when the complete output
    still fits.
    """

    if not documents:
        return ()
    query_terms = _terms(query)
    if not query_terms:
        return ()
    sentences = _score_documents(documents, query_terms)
    if not sentences:
        return ()
    chosen: dict[int, set[int]] = {}
    for sentence in _selection_order(sentences):
        picked = chosen.setdefault(sentence.document, set())
        if sentence.index in picked:
            continue
        trial = {key: set(value) for key, value in chosen.items()}
        trial.setdefault(sentence.document, set()).add(sentence.index)
        fragments = _fragments(documents, trial)
        if measure(fragments) > max_bytes:
            continue
        chosen = trial
    fragments = _fragments(documents, chosen)
    return tuple(_ordered(fragments, documents))


def _terms(value: str) -> tuple[str, ...]:
    return tuple(_stem(term) for term, _start, _end in analyze_text_with_spans(value))


def _stem(term: str) -> str:
    """Fold the most common English inflections so a claim matches its evidence."""

    for suffix in _STEM_SUFFIXES:
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def _sentences(text: str) -> tuple[tuple[int, int], ...]:
    """Return the non-empty sentence spans of one body, in reading order."""

    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        spans.extend(_trimmed_span(text, start, match.start()))
        start = match.end()
    spans.extend(_trimmed_span(text, start, len(text)))
    return tuple(spans)


def _trimmed_span(text: str, start: int, end: int) -> tuple[tuple[int, int], ...]:
    chunk = text[start:end]
    stripped = chunk.strip()
    if not stripped:
        return ()
    offset = start + (len(chunk) - len(chunk.lstrip()))
    return ((offset, offset + len(stripped)),)


def _score_documents(
    documents: Sequence[EvidenceDocument],
    query_terms: Sequence[str],
) -> tuple[_Sentence, ...]:
    """Score every sentence of every document against the query terms.

    The term weights are the inverse sentence frequency over the recalled
    bodies, so a term that only one sentence carries counts for much more
    than a term every sentence repeats. A sentence that neighbours other
    matching sentences is preferred, because the evidence of one claim is
    usually a coherent passage.
    """

    spans: list[tuple[int, int, int, str, list[str]]] = []
    for document_index, document in enumerate(documents):
        for start, end in _sentences(document.text):
            spans.append(
                (document_index, start, end, document.text[start:end], _terms(document.text[start:end]))
            )
    if not spans:
        return ()
    total = len(spans)
    weights: dict[str, float] = {}
    for term in set(query_terms):
        frequency = sum(1 for span in spans if term in span[4])
        weights[term] = math.log((total + 1) / (frequency + 1)) + 1.0
    grouped: dict[int, list[int]] = {}
    for position, span in enumerate(spans):
        grouped.setdefault(span[0], []).append(position)
    smoothed: list[_Sentence] = []
    for positions in grouped.values():
        raw = [_sentence_score(span[4], weights) for span in (spans[position] for position in positions)]
        if raw and raw[0] > 0:
            raw[0] -= _LEADING_SENTENCE_DEMOTION * max(raw)
        for offset, position in enumerate(positions):
            document_index, start, end, text, _span_terms = spans[position]
            left = raw[offset - 1] if offset > 0 else 0.0
            right = raw[offset + 1] if offset + 1 < len(raw) else 0.0
            smoothed.append(
                _Sentence(
                    document=document_index,
                    start=start,
                    end=end,
                    text=text,
                    score=raw[offset] + _NEIGHBOUR_WEIGHT * (left + right),
                    index=offset,
                )
            )
    return tuple(smoothed)


def _sentence_score(terms: Sequence[str], weights: dict[str, float]) -> float:
    counts: dict[str, int] = {}
    for term in terms:
        if term in weights:
            counts[term] = counts.get(term, 0) + 1
    return sum(weights[term] * (1.0 + math.log(count)) for term, count in counts.items())


def _selection_order(sentences: Sequence[_Sentence]) -> tuple[_Sentence, ...]:
    """Order the candidates so every document gets a chance before one gets more.

    A request usually offers several documents and the evidence lives in one
    of them. Handing the first pick to each document therefore keeps a strong
    candidate from consuming the whole budget on its own.
    """

    by_document: dict[int, list[_Sentence]] = {}
    for sentence in sentences:
        by_document.setdefault(sentence.document, []).append(sentence)
    for group in by_document.values():
        group.sort(key=lambda item: (-item.score, item.index))
    document_order = sorted(by_document, key=lambda index: (-by_document[index][0].score, index))
    ordered: list[_Sentence] = []
    depth = 0
    while True:
        added = False
        for index in document_order:
            group = by_document[index]
            if depth < len(group):
                ordered.append(group[depth])
                added = True
        if not added:
            break
        depth += 1
    return tuple(ordered)


def _fragments(
    documents: Sequence[EvidenceDocument],
    chosen: dict[int, set[int]],
) -> tuple[SelectedFragment, ...]:
    """Merge the chosen sentences of every document into contiguous fragments.

    Neighbouring sentences become one fragment, so the caller pays for one
    citation instead of one per sentence and the byte budget reaches further
    into the passage.
    """

    fragments: list[SelectedFragment] = []
    for index, document in enumerate(documents):
        picked = chosen.get(index)
        if not picked:
            continue
        spans = _sentences(document.text)
        runs: list[tuple[int, int]] = []
        for position in sorted(picked):
            if runs and position == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], position)
            else:
                runs.append((position, position))
        for first, last in runs:
            start = spans[first][0]
            end = spans[last][1]
            fragments.append(
                SelectedFragment(
                    scope_id=document.scope_id,
                    memory_ref=document.memory_ref,
                    entry_id=document.entry_id,
                    entry_version_id=document.entry_version_id,
                    start=start,
                    end=end,
                    text=document.text[start:end],
                )
            )
    return tuple(fragments)


def _ordered(
    fragments: Sequence[SelectedFragment],
    documents: Sequence[EvidenceDocument],
) -> list[SelectedFragment]:
    """Return the selected fragments in reading order."""

    order = {
        (document.scope_id, document.entry_id, document.entry_version_id): index
        for index, document in enumerate(documents)
    }
    return sorted(
        fragments,
        key=lambda fragment: (
            order.get((fragment.scope_id, fragment.entry_id, fragment.entry_version_id), 0),
            fragment.start,
        ),
    )


__all__ = [
    "EvidenceDocument",
    "SelectedFragment",
    "select_evidence_fragments",
]
