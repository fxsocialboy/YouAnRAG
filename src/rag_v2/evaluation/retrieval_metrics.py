"""Pure retrieval metrics used by the final Legacy-vs-V2 evaluation."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def recall_at_k(ranked_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(_unique(relevant_ids))
    if not relevant:
        return None
    retrieved = set(_unique(ranked_ids)[:k])
    return len(relevant & retrieved) / len(relevant)


def reciprocal_rank_at_k(ranked_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(_unique(relevant_ids))
    if not relevant:
        return None
    for rank, item_id in enumerate(_unique(ranked_ids)[:k], 1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(_unique(relevant_ids))
    if not relevant:
        return None
    ranked = _unique(ranked_ids)[:k]
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, item_id in enumerate(ranked, 1) if item_id in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ranked_results(
    results: list[dict[str, Any]],
    *,
    relevant_source_files: list[str],
    relevant_chunk_ids: list[str],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float | None]:
    ranked_docs = _unique(str(item.get("source_file", "")) for item in results if item.get("source_file"))
    ranked_chunks = _unique(str(item.get("chunk_id", "")) for item in results if item.get("chunk_id"))
    metrics: dict[str, float | None] = {}
    for k in ks:
        metrics[f"doc_recall@{k}"] = recall_at_k(ranked_docs, relevant_source_files, k)
        metrics[f"chunk_recall@{k}"] = recall_at_k(ranked_chunks, relevant_chunk_ids, k)
    max_k = max(ks)
    metrics[f"doc_mrr@{max_k}"] = reciprocal_rank_at_k(ranked_docs, relevant_source_files, max_k)
    metrics[f"chunk_mrr@{max_k}"] = reciprocal_rank_at_k(ranked_chunks, relevant_chunk_ids, max_k)
    metrics[f"doc_ndcg@{max_k}"] = ndcg_at_k(ranked_docs, relevant_source_files, max_k)
    metrics[f"chunk_ndcg@{max_k}"] = ndcg_at_k(ranked_chunks, relevant_chunk_ids, max_k)
    return metrics


def summarize_retrieval_rows(rows: list[dict[str, float | int | None]]) -> dict[str, float | int | None]:
    keys = sorted({key for row in rows for key in row if key not in {"latency_ms"}})
    summary: dict[str, float | int | None] = {"query_count": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[key] = round(statistics.mean(values), 4) if values else None
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    if latencies:
        ordered = sorted(latencies)
        summary.update(
            {
                "avg_latency_ms": round(statistics.mean(ordered), 2),
                "p50_latency_ms": round(_percentile(ordered, 0.50), 2),
                "p95_latency_ms": round(_percentile(ordered, 0.95), 2),
            }
        )
    return summary


def _percentile(ordered: list[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
