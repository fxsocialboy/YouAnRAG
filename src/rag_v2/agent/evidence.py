"""Evidence construction utilities for Stage6 Agent answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_v2.agent.models import EvidenceItem


@dataclass(slots=True)
class EvidenceBuilder:
    """Convert retrieval/context objects into ranked, citation-ready evidence.

    Citation ids are local to one answer and assigned by rank as S1, S2 ... .
    Duplicate chunks are removed before numbering so the same chunk appears only
    once in the answer source list.
    """

    max_items: int | None = None

    def build_from_search_results(self, results: list[Any]) -> list[EvidenceItem]:
        candidates = [_search_result_to_candidate(item) for item in results]
        return self._rank_and_number(candidates)

    def build_from_context_chunks(self, chunks: list[Any]) -> list[EvidenceItem]:
        candidates = [_context_chunk_to_candidate(item) for item in chunks]
        return self._rank_and_number(candidates)

    def _rank_and_number(self, candidates: list[dict[str, Any]]) -> list[EvidenceItem]:
        deduped: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            chunk_id = str(candidate.get("chunk_id", ""))
            if not chunk_id:
                continue
            existing = deduped.get(chunk_id)
            if existing is None or _sort_score(candidate) > _sort_score(existing):
                deduped[chunk_id] = candidate

        ranked = sorted(deduped.values(), key=lambda item: (-_sort_score(item), int(item.get("original_rank", 10**9)), item["chunk_id"]))
        if self.max_items is not None:
            ranked = ranked[: self.max_items]

        evidence: list[EvidenceItem] = []
        for idx, item in enumerate(ranked, 1):
            confidence, confidence_type = _retrieval_confidence(item)
            metadata = dict(item.get("metadata", {}) or {})
            metadata.update(
                {
                    "sort_score": _sort_score(item),
                    "retrieval_confidence": confidence,
                    "confidence_type": confidence_type,
                    "confidence_state": _confidence_state(confidence),
                }
            )
            evidence.append(
                EvidenceItem(
                    citation_id=f"S{idx}",
                    chunk_id=str(item["chunk_id"]),
                    source_file=str(item["source_file"]),
                    section_path=list(item.get("section_path", [])),
                    content=str(item.get("content", "")),
                    # ``score`` remains the public compatibility field, but it
                    # now means retrieval confidence—not MMR selection score.
                    score=float(confidence or 0.0),
                    rank=idx,
                    metadata=metadata,
                )
            )
        return evidence


def _sort_score(item: dict[str, Any]) -> float:
    score = item.get("sort_score", item.get("score", 0.0))
    try:
        return float(score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _search_result_to_candidate(item: Any) -> dict[str, Any]:
    section_path_text = str(getattr(item, "section_path_text", ""))
    section_path = [part.strip() for part in section_path_text.split("/") if part.strip()]
    metadata = dict(getattr(item, "metadata", {}) or {})
    fusion_score = float(getattr(item, "fusion_score", 0.0) or 0.0)
    sort_score = _first_optional_score(
        getattr(item, "mmr_score", None),
        getattr(item, "rerank_score", None),
        fusion_score,
    )
    return {
        "chunk_id": str(getattr(item, "chunk_id", "")),
        "source_file": str(getattr(item, "source_file", "")),
        "section_path": section_path,
        "content": str(getattr(item, "content", "")),
        "score": sort_score,
        "sort_score": sort_score,
        "original_rank": int(getattr(item, "rank", 0) or 0),
        "metadata": {
            "dense_score": getattr(item, "dense_score", None),
            "sparse_score": getattr(item, "sparse_score", None),
            "fusion_score": fusion_score,
            "rerank_score": getattr(item, "rerank_score", None),
            "mmr_score": getattr(item, "mmr_score", None),
            "sort_score": sort_score,
            **metadata,
        },
    }


def _context_chunk_to_candidate(item: Any) -> dict[str, Any]:
    chunk_ids = list(getattr(item, "chunk_ids", []) or [])
    chunk_indexes = list(getattr(item, "chunk_indexes", []) or [])
    section_path_text = str(getattr(item, "section_path_text", ""))
    metadata = dict(getattr(item, "metadata", {}) or {})
    fusion_score = float(getattr(item, "fusion_score", 0.0) or 0.0)
    sort_score = _first_optional_score(
        metadata.get("mmr_score"),
        metadata.get("rerank_score"),
        fusion_score,
    )
    return {
        "chunk_id": str(chunk_ids[0] if chunk_ids else f"context::{id(item)}"),
        "source_file": str(getattr(item, "source_file", "")),
        "section_path": [section_path_text] if section_path_text else [],
        "content": str(getattr(item, "content", "")),
        "score": sort_score,
        "sort_score": sort_score,
        "original_rank": int(getattr(item, "rank", 0) or 0),
        "metadata": {
            "chunk_ids": chunk_ids,
            "chunk_indexes": chunk_indexes,
            "token_count": getattr(item, "token_count", None),
            "fusion_score": fusion_score,
            **metadata,
        },
    }


def _first_optional_score(*scores: Any) -> float:
    for score in scores:
        if score is None:
            continue
        try:
            return float(score)
        except (TypeError, ValueError):
            continue
    return 0.0


def _retrieval_confidence(item: dict[str, Any]) -> tuple[float | None, str]:
    """Return an absolute relevance signal, never an MMR/RRF relative score."""

    metadata = dict(item.get("metadata", {}) or {})
    for key, confidence_type in (("rerank_score", "rerank"), ("dense_score", "dense")):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value), confidence_type
        except (TypeError, ValueError):
            continue
    return None, "unavailable"


def _confidence_state(confidence: float | None) -> str:
    if confidence is None:
        return "confidence_unavailable"
    if confidence < 0.20:
        return "low_confidence"
    if confidence < 0.50:
        return "uncertain"
    return "retrieval_confident"
