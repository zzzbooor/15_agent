from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any

from .tokenizer import display_token, normalize_text, tokenize


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("search ranking deadline exceeded")


def rank_documents(
    query: str,
    documents: list[dict[str, Any]],
    *,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    """Rank in-memory documents with BM25 plus phrase/title boosts."""

    if not documents:
        return []
    _check_deadline(deadline)
    query_tokens = tokenize(query)
    query_counter = Counter(query_tokens)
    doc_tokens = [document["tokens"] for document in documents]
    doc_counts = []
    for tokens in doc_tokens:
        _check_deadline(deadline)
        doc_counts.append(Counter(tokens))
    average_length = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens) or 1.0
    document_frequency = {
        term: sum(1 for counts in doc_counts if counts.get(term, 0) > 0)
        for term in query_counter
    }
    compact_query = _compact(query)
    ranked: list[dict[str, Any]] = []
    k1 = 1.5
    b = 0.75
    for index, document in enumerate(documents):
        _check_deadline(deadline)
        counts = doc_counts[index]
        length = len(doc_tokens[index]) or 1
        score = 0.0
        matched: set[str] = set()
        for term, query_frequency in query_counter.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            matched.add(display_token(term))
            df = document_frequency[term]
            inverse_document_frequency = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * length / average_length)
            score += query_frequency * inverse_document_frequency * frequency * (k1 + 1.0) / denominator

        compact_text = _compact(document["text"])
        compact_filename = _compact(document["filename"])
        phrase_match = bool(compact_query and compact_query in compact_text)
        filename_match = bool(compact_query and compact_query in compact_filename)
        if phrase_match:
            score += 4.0
        if filename_match:
            score += 2.0
        elif any(token in document["filename_tokens"] for token in query_counter):
            score += 0.5
        if score <= 0:
            continue
        ranked.append(
            {
                "document_index": index,
                "score": score,
                "matched_terms": sorted(matched),
                "phrase_match": phrase_match,
                "filename_match": filename_match,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], documents[item["document_index"]]["relative_path"]))
    return ranked
