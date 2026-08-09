"""Hybrid dense+sparse retrieval with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rag_v2.retrieval.bm25_index import BM25Index, BM25SearchHit
from rag_v2.retrieval.qdrant_searcher import QdrantSearcher, Stage2SearchResult


class DenseSearcher(Protocol):
    def search(self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None) -> list[Stage2SearchResult]: ...


class SparseSearcher(Protocol):
    def search(self, query: str, top_k: int = 10, source_file: str | None = None) -> list[BM25SearchHit]: ...


@dataclass(slots=True)
class HybridSearchResult:
    chunk_id: str
    content: str
    source_file: str
    chunk_index: int
    section_path_text: str
    dense_score: float | None
    sparse_score: float | None
    fusion_score: float
    rank: int
    metadata: dict[str, Any]
    rerank_score: float | None = None
    mmr_score: float | None = None
    stage4_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "section_path_text": self.section_path_text,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "fusion_score": self.fusion_score,
            "rank": self.rank,
            "rerank_score": self.rerank_score,
            "mmr_score": self.mmr_score,
            "stage4_rank": self.stage4_rank,
            "metadata": self.metadata,
        }


class HybridSearcher:
    """Fuse Stage2 Qdrant dense retrieval and Stage3 BM25 sparse retrieval."""

    def __init__(
        self,
        dense_searcher: DenseSearcher,
        sparse_index: SparseSearcher,
        *,
        rrf_k: int = 60,
    ):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.dense_searcher = dense_searcher
        self.sparse_index = sparse_index
        self.rrf_k = rrf_k

    @classmethod
    def from_config(
        cls,
        *,
        dense_searcher: QdrantSearcher,
        sparse_index: BM25Index,
        rrf_k: int = 60,
    ) -> "HybridSearcher":
        return cls(dense_searcher=dense_searcher, sparse_index=sparse_index, rrf_k=rrf_k)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridSearchResult]:
        if top_k <= 0:
            return []

        source_file = str(filters.get("source_file")) if filters and filters.get("source_file") else None
        dense_hits = self.dense_searcher.search(query, top_k=dense_top_k, filters=filters)
        sparse_hits = self.sparse_index.search(query, top_k=sparse_top_k, source_file=source_file)

        fused: dict[str, dict[str, Any]] = {}

        for hit in dense_hits:
            entry = fused.setdefault(
                hit.chunk_id,
                {
                    "chunk_id": hit.chunk_id,
                    "content": hit.content,
                    "source_file": hit.source_file,
                    "chunk_index": hit.chunk_index,
                    "section_path_text": hit.section_path_text,
                    "dense_score": None,
                    "sparse_score": None,
                    "fusion_score": 0.0,
                    "metadata": {
                        "section_path": hit.section_path,
                        "content_hash": hit.content_hash,
                        "is_active": hit.is_active,
                        "point_id": hit.point_id,
                        "content_preview": hit.content_preview,
                        "token_count": hit.token_count,
                    },
                },
            )
            entry["dense_score"] = hit.score
            entry["fusion_score"] += reciprocal_rank_score(hit.rank, self.rrf_k)

        for hit in sparse_hits:
            entry = fused.setdefault(
                hit.chunk_id,
                {
                    "chunk_id": hit.chunk_id,
                    "content": hit.content,
                    "source_file": hit.source_file,
                    "chunk_index": hit.chunk_index,
                    "section_path_text": hit.section_path_text,
                    "dense_score": None,
                    "sparse_score": None,
                    "fusion_score": 0.0,
                    "metadata": {
                        "matched_terms": hit.matched_terms,
                        "content_preview": hit.content_preview,
                        "token_count": hit.token_count,
                    },
                },
            )
            entry["sparse_score"] = hit.score
            entry["fusion_score"] += reciprocal_rank_score(hit.rank, self.rrf_k)
            entry["metadata"].setdefault("matched_terms", hit.matched_terms)
            entry["metadata"].setdefault("content_preview", hit.content_preview)
            entry["metadata"].setdefault("token_count", hit.token_count)

        ranked = sorted(
            fused.values(),
            key=lambda item: (-float(item["fusion_score"]), item["chunk_id"]),
        )
        results: list[HybridSearchResult] = []
        for rank, item in enumerate(ranked[:top_k], 1):
            results.append(
                HybridSearchResult(
                    chunk_id=str(item["chunk_id"]),
                    content=str(item["content"]),
                    source_file=str(item["source_file"]),
                    chunk_index=int(item["chunk_index"]),
                    section_path_text=str(item["section_path_text"]),
                    dense_score=float(item["dense_score"]) if item["dense_score"] is not None else None,
                    sparse_score=float(item["sparse_score"]) if item["sparse_score"] is not None else None,
                    fusion_score=round(float(item["fusion_score"]), 6),
                    rank=rank,
                    metadata=dict(item["metadata"]),
                )
            )
        return results

    def search_dicts(
        self,
        query: str,
        *,
        top_k: int = 10,
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            item.to_dict()
            for item in self.search(
                query,
                top_k=top_k,
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                filters=filters,
            )
        ]

    def invoke(
        self,
        query: str,
        *,
        top_k: int = 10,
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        filters: dict[str, Any] | None = None,
    ) -> list[str]:
        return [
            item.content
            for item in self.search(
                query,
                top_k=top_k,
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                filters=filters,
            )
        ]


def reciprocal_rank_score(rank: int, k: int = 60) -> float:
    if rank <= 0:
        return 0.0
    return 1.0 / (k + rank)
