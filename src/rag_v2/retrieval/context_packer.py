"""Context packing for Stage3 evidence assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from rag_v2.retrieval.hybrid_searcher import HybridSearchResult


@dataclass(slots=True)
class ContextChunk:
    citation_id: str
    source_file: str
    section_path_text: str
    chunk_ids: list[str]
    chunk_indexes: list[int]
    content: str
    token_count: int
    fusion_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "source_file": self.source_file,
            "section_path_text": self.section_path_text,
            "chunk_ids": self.chunk_ids,
            "chunk_indexes": self.chunk_indexes,
            "content": self.content,
            "token_count": self.token_count,
            "fusion_score": self.fusion_score,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ContextPackResult:
    evidence_chunks: list[ContextChunk]
    total_tokens: int
    token_budget: int
    insufficient_evidence: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_chunks": [item.to_dict() for item in self.evidence_chunks],
            "total_tokens": self.total_tokens,
            "token_budget": self.token_budget,
            "insufficient_evidence": self.insufficient_evidence,
        }


class ContextPacker:
    """Pack hybrid retrieval results into citation-ready evidence chunks."""

    def __init__(self, metadata_rows: list[dict[str, Any]]):
        self.metadata_rows = metadata_rows
        self.by_chunk_id: dict[str, dict[str, Any]] = {}
        self.by_source_and_index: dict[str, dict[int, dict[str, Any]]] = {}
        self._build_indexes()

    @classmethod
    def from_metadata_path(cls, path: str | Path) -> "ContextPacker":
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(rows)

    def pack(
        self,
        candidates: list[HybridSearchResult],
        *,
        token_budget: int = 1200,
        same_section_extra: int = 1,
        min_evidence_chunks: int = 1,
    ) -> ContextPackResult:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        expanded = self._expand_candidates(candidates, same_section_extra=same_section_extra)
        merged = self._merge_adjacent(expanded)
        ranked = sorted(merged, key=lambda item: (-float(item["sort_score"]), item["source_file"], item["chunk_indexes"][0]))

        selected: list[ContextChunk] = []
        total_tokens = 0
        for idx, item in enumerate(ranked, 1):
            if total_tokens + int(item["token_count"]) > token_budget:
                continue
            citation_id = f"[S{len(selected) + 1}]"
            selected.append(
                ContextChunk(
                    citation_id=citation_id,
                    source_file=str(item["source_file"]),
                    section_path_text=str(item["section_path_text"]),
                    chunk_ids=list(item["chunk_ids"]),
                    chunk_indexes=list(item["chunk_indexes"]),
                    content=str(item["content"]),
                    token_count=int(item["token_count"]),
                    fusion_score=round(float(item["fusion_score"]), 6),
                    metadata=dict(item["metadata"]),
                )
            )
            total_tokens += int(item["token_count"])
            if total_tokens >= token_budget:
                break

        insufficient = len(selected) < min_evidence_chunks
        return ContextPackResult(
            evidence_chunks=selected,
            total_tokens=total_tokens,
            token_budget=token_budget,
            insufficient_evidence=insufficient,
        )

    def _build_indexes(self) -> None:
        for row in self.metadata_rows:
            chunk_id = str(row.get("chunk_id", ""))
            source_file = str(row.get("source_file", ""))
            chunk_index = int(row.get("chunk_index", -1))
            if not chunk_id or not source_file or chunk_index < 0:
                continue
            self.by_chunk_id[chunk_id] = row
            self.by_source_and_index.setdefault(source_file, {})[chunk_index] = row

    def _expand_candidates(
        self,
        candidates: list[HybridSearchResult],
        *,
        same_section_extra: int,
    ) -> list[dict[str, Any]]:
        expanded: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            self._add_row(
                expanded,
                candidate.chunk_id,
                candidate.fusion_score,
                candidate.rerank_score,
                candidate.mmr_score,
                candidate.dense_score,
            )
            self._add_row(
                expanded,
                self._neighbor_chunk_id(candidate.source_file, candidate.chunk_index - 1, candidate.section_path_text),
                candidate.fusion_score * 0.97,
                _decay_optional_score(candidate.rerank_score, 0.97),
                _decay_optional_score(candidate.mmr_score, 0.97),
                _decay_optional_score(candidate.dense_score, 0.97),
            )
            self._add_row(
                expanded,
                self._neighbor_chunk_id(candidate.source_file, candidate.chunk_index + 1, candidate.section_path_text),
                candidate.fusion_score * 0.97,
                _decay_optional_score(candidate.rerank_score, 0.97),
                _decay_optional_score(candidate.mmr_score, 0.97),
                _decay_optional_score(candidate.dense_score, 0.97),
            )

            if same_section_extra > 0:
                same_section_rows = self._same_section_neighbors(candidate.source_file, candidate.chunk_id, candidate.section_path_text)
                for row in same_section_rows[:same_section_extra]:
                    self._add_row(
                        expanded,
                        str(row.get("chunk_id")),
                        candidate.fusion_score * 0.94,
                        _decay_optional_score(candidate.rerank_score, 0.94),
                        _decay_optional_score(candidate.mmr_score, 0.94),
                        _decay_optional_score(candidate.dense_score, 0.94),
                    )
        return list(expanded.values())

    def _add_row(
        self,
        expanded: dict[str, dict[str, Any]],
        chunk_id: str | None,
        fusion_score: float,
        rerank_score: float | None = None,
        mmr_score: float | None = None,
        dense_score: float | None = None,
    ) -> None:
        if not chunk_id:
            return
        row = self.by_chunk_id.get(chunk_id)
        if not row:
            return
        sort_score = _stage4_sort_score(fusion_score, rerank_score, mmr_score)
        existing = expanded.get(chunk_id)
        if existing:
            existing["fusion_score"] = max(float(existing["fusion_score"]), float(fusion_score))
            existing["sort_score"] = max(float(existing["sort_score"]), float(sort_score))
            existing["rerank_score"] = _max_optional_score(existing.get("rerank_score"), rerank_score)
            existing["mmr_score"] = _max_optional_score(existing.get("mmr_score"), mmr_score)
            existing["dense_score"] = _max_optional_score(existing.get("dense_score"), dense_score)
            existing["metadata"]["rerank_score"] = existing["rerank_score"]
            existing["metadata"]["mmr_score"] = existing["mmr_score"]
            existing["metadata"]["sort_score"] = existing["sort_score"]
            existing["metadata"]["dense_score"] = existing["dense_score"]
            return
        expanded[chunk_id] = {
            "chunk_id": chunk_id,
            "source_file": str(row.get("source_file", "")),
            "section_path_text": str(row.get("section_path_text") or row.get("metadata", {}).get("section_path_text", "")),
            "chunk_indexes": [int(row.get("chunk_index", -1))],
            "chunk_ids": [chunk_id],
            "content": str(row.get("content", "")),
            "token_count": int(row.get("token_count", 0)),
            "fusion_score": float(fusion_score),
            "rerank_score": rerank_score,
            "mmr_score": mmr_score,
            "dense_score": dense_score,
            "sort_score": sort_score,
            "metadata": {
                "content_hashes": [str(row.get("content_hash", ""))],
                "content_preview": str(row.get("content", ""))[:160],
                "rerank_score": rerank_score,
                "mmr_score": mmr_score,
                "sort_score": sort_score,
                "dense_score": dense_score,
            },
        }

    def _neighbor_chunk_id(self, source_file: str, chunk_index: int, section_path_text: str) -> str | None:
        row = self.by_source_and_index.get(source_file, {}).get(chunk_index)
        if not row:
            return None
        row_section = str(row.get("section_path_text") or row.get("metadata", {}).get("section_path_text", ""))
        if row_section != section_path_text:
            return None
        return str(row.get("chunk_id"))

    def _same_section_neighbors(self, source_file: str, chunk_id: str, section_path_text: str) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.by_source_and_index.get(source_file, {}).values()
            if str(row.get("section_path_text") or row.get("metadata", {}).get("section_path_text", "")) == section_path_text
            and str(row.get("chunk_id")) != chunk_id
        ]
        rows.sort(key=lambda item: int(item.get("chunk_index", 10**9)))
        return rows

    def _merge_adjacent(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = sorted(rows, key=lambda item: (item["source_file"], item["chunk_indexes"][0]))
        merged: list[dict[str, Any]] = []
        for row in rows:
            if not merged:
                merged.append(row)
                continue
            prev = merged[-1]
            can_merge = (
                prev["source_file"] == row["source_file"]
                and prev["section_path_text"] == row["section_path_text"]
                and prev["chunk_indexes"][-1] + 1 == row["chunk_indexes"][0]
            )
            if not can_merge:
                merged.append(row)
                continue
            prev["chunk_indexes"].extend(row["chunk_indexes"])
            prev["chunk_ids"].extend(row["chunk_ids"])
            prev["content"] = f"{prev['content']}\n{row['content']}".strip()
            prev["token_count"] = int(prev["token_count"]) + int(row["token_count"])
            prev["fusion_score"] = max(float(prev["fusion_score"]), float(row["fusion_score"]))
            prev["sort_score"] = max(float(prev["sort_score"]), float(row["sort_score"]))
            prev["rerank_score"] = _max_optional_score(prev.get("rerank_score"), row.get("rerank_score"))
            prev["mmr_score"] = _max_optional_score(prev.get("mmr_score"), row.get("mmr_score"))
            prev["dense_score"] = _max_optional_score(prev.get("dense_score"), row.get("dense_score"))
            prev["metadata"]["content_hashes"].extend(row["metadata"].get("content_hashes", []))
            prev["metadata"]["rerank_score"] = prev["rerank_score"]
            prev["metadata"]["mmr_score"] = prev["mmr_score"]
            prev["metadata"]["sort_score"] = prev["sort_score"]
            prev["metadata"]["dense_score"] = prev["dense_score"]
        return merged


def _stage4_sort_score(fusion_score: float, rerank_score: float | None = None, mmr_score: float | None = None) -> float:
    if mmr_score is not None:
        return float(mmr_score)
    if rerank_score is not None:
        return float(rerank_score)
    return float(fusion_score)


def _decay_optional_score(score: float | None, decay: float) -> float | None:
    if score is None:
        return None
    return float(score) * decay


def _max_optional_score(left: Any, right: Any) -> float | None:
    values = [float(value) for value in (left, right) if value is not None]
    return max(values) if values else None
