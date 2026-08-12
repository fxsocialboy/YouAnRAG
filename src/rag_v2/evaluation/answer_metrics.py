"""Deterministic answer/citation metrics for Stage7.

Faithfulness and relevancy values are supplied later by the DeepSeek judge;
this module only validates and aggregates them.  Citation mapping and fallback
accuracy remain deterministic and independent from an LLM judge.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Iterable

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def citation_mapping_metrics(answer: str, known_citation_ids: Iterable[str]) -> dict[str, Any]:
    known = {str(item).strip("[] ") for item in known_citation_ids}
    used = list(dict.fromkeys(CITATION_PATTERN.findall(str(answer))))
    mapped = [item for item in used if item in known]
    unknown = [item for item in used if item not in known]
    return {
        "used_citation_ids": used,
        "mapped_citation_ids": mapped,
        "unknown_citation_ids": unknown,
        "citation_count": len(used),
        "mapped_count": len(mapped),
        "all_citations_mapped": not unknown,
        "citation_mapping_ratio": len(mapped) / len(used) if used else 1.0,
    }


def fallback_accuracy(rows: Iterable[dict[str, Any]]) -> float | None:
    labeled = [row for row in rows if row.get("expected_fallback") is not None]
    if not labeled:
        return None
    return sum(bool(row["expected_fallback"]) == bool(row.get("is_fallback")) for row in labeled) / len(labeled)


def latency_summary(latencies_ms: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in latencies_ms)
    if not ordered:
        return {"avg_latency_ms": 0.0, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0}
    return {
        "avg_latency_ms": round(statistics.mean(ordered), 2),
        "p50_latency_ms": round(_percentile(ordered, 0.50), 2),
        "p95_latency_ms": round(_percentile(ordered, 0.95), 2),
    }


def summarize_answer_rows(rows: list[dict[str, Any]], *, composer_mode: str | None = None) -> dict[str, Any]:
    selected = [row for row in rows if composer_mode is None or row.get("composer_mode") == composer_mode]
    summary: dict[str, Any] = {
        "query_count": len(selected),
        "composer_mode": composer_mode or "mixed",
        "answer_success_rate": _ratio(selected, lambda row: not row.get("error")),
        "fallback_rate": _ratio(selected, lambda row: bool(row.get("is_fallback"))),
        "all_citations_mapped_ratio": _ratio(selected, lambda row: bool(row.get("all_citations_mapped", False))),
        "fallback_accuracy": fallback_accuracy(selected),
        "fallback_reason_consistency": _ratio(
            selected,
            lambda row: bool(row.get("fallback_reason")) == bool(row.get("is_fallback")),
        ),
        "judge_coverage": _ratio(
            [row for row in selected if not row.get("is_fallback")],
            lambda row: row.get("judge") is not None,
        ),
    }
    for metric in ("faithfulness", "answer_relevancy", "citation_correctness", "citation_completeness"):
        values = [
            float(row[metric])
            for row in selected
            if not row.get("is_fallback") and row.get(metric) is not None
        ]
        summary[metric] = round(statistics.mean(values), 4) if values else None
    summary.update(latency_summary(row["latency_ms"] for row in selected if row.get("latency_ms") is not None))
    return summary


def _ratio(rows: list[dict[str, Any]], predicate) -> float:
    return round(sum(bool(predicate(row)) for row in rows) / len(rows), 4) if rows else 0.0


def _percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
