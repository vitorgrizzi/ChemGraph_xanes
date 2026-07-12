"""Temporal evaluation helpers for literature-KG missing-link predictions."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from chemgraph.kg.extract import extract_records_from_chunks
from chemgraph.kg.schema import CatalystRecord, PaperChunk
from chemgraph.kg.verify import verify_records


def year_from_chunk(chunk: PaperChunk) -> int | None:
    for value in (
        chunk.metadata.get("year"),
        chunk.metadata.get("publication_year"),
        chunk.source_path or "",
        chunk.text[:500],
    ):
        match = re.search(r"\b(19|20)\d{2}\b", str(value))
        if match:
            return int(match.group(0))
    return None


def _verified_records(chunks: Iterable[PaperChunk]) -> list[CatalystRecord]:
    records = extract_records_from_chunks(chunks)
    return [result.record for result in verify_records(records) if result.accepted]


def _pair(record: CatalystRecord) -> tuple[str, str] | None:
    if not record.active_metals or not record.support:
        return None
    return record.active_metals[0], record.support


def temporal_link_backtest(
    chunks: list[PaperChunk],
    *,
    split_year: int,
    top_k: int = 10,
) -> dict[str, Any]:
    """Predict unseen active-metal/support pairs and score future recovery."""
    before_chunks = [chunk for chunk in chunks if (year_from_chunk(chunk) or 10**9) <= split_year]
    after_chunks = [chunk for chunk in chunks if (year_from_chunk(chunk) or 0) > split_year]
    unknown = sum(year_from_chunk(chunk) is None for chunk in chunks)
    before_records = _verified_records(before_chunks)
    after_records = _verified_records(after_chunks)

    observed_before = {pair for record in before_records if (pair := _pair(record))}
    observed_after = {pair for record in after_records if (pair := _pair(record))}
    future_new = observed_after - observed_before
    metal_scores: dict[str, list[float]] = defaultdict(list)
    support_scores: dict[str, list[float]] = defaultdict(list)
    for record in before_records:
        pair = _pair(record)
        if not pair:
            continue
        values = [metric.value for metric in record.performance_metrics if metric.value is not None]
        score = max(values) if values else record.confidence
        metal_scores[pair[0]].append(float(score))
        support_scores[pair[1]].append(float(score))

    candidates = []
    for metal, metal_values in metal_scores.items():
        for support, support_values in support_scores.items():
            pair = (metal, support)
            if pair in observed_before:
                continue
            score = 0.5 * max(metal_values) + 0.5 * max(support_values)
            candidates.append((score, pair))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    predicted = [pair for _, pair in candidates[:top_k]]
    hits = [pair for pair in predicted if pair in future_new]
    precision_at_k = len(hits) / len(predicted) if predicted else 0.0
    recall_at_k = len(hits) / len(future_new) if future_new else 0.0
    reciprocal_rank = 0.0
    for rank, pair in enumerate(predicted, start=1):
        if pair in future_new:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "ok": True,
        "task": "active_metal_support_link_prediction",
        "split_year": split_year,
        "chunks_before_or_equal_split": len(before_chunks),
        "chunks_after_split": len(after_chunks),
        "chunks_without_year": unknown,
        "records_before": len(before_records),
        "records_after": len(after_records),
        "future_new_links": [list(pair) for pair in sorted(future_new)],
        "predictions": [
            {"active_metal": pair[0], "support": pair[1], "score": score}
            for score, pair in candidates[:top_k]
        ],
        "hits": [list(pair) for pair in hits],
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mean_reciprocal_rank": reciprocal_rank,
    }
