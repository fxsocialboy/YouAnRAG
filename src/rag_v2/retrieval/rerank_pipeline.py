"""Stage4 rerank pipeline orchestration.

This module composes Stage3 hybrid retrieval with optional Cross-Encoder rerank
and optional MMR diversity selection. It deliberately keeps the output type as
``list[HybridSearchResult]`` so downstream ContextPacker can consume it directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from rag_v2.retrieval.hybrid_searcher import HybridSearchResult, HybridSearcher
from rag_v2.retrieval.mmr import MMRSelector
from rag_v2.retrieval.reranker import BaseReranker


class HybridRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridSearchResult]: ...


class DiversitySelector(Protocol):
    def select(self, candidates: list[HybridSearchResult], top_k: int) -> list[HybridSearchResult]: ...


@dataclass(slots=True)
class RerankPipelineOptions:
    top_k: int = 10
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rerank_top_k: int = 30
    mmr_pre_candidates: int = 20
    mmr_top_k: int = 10
    enable_reranker: bool = True
    enable_mmr: bool = True

    def validate(self) -> None:
        for name in ("top_k", "dense_top_k", "sparse_top_k", "rerank_top_k", "mmr_pre_candidates", "mmr_top_k"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RerankPipeline:
    """Compose hybrid retrieval, optional reranker and optional MMR."""

    def __init__(
        self,
        hybrid_searcher: HybridRetriever,
        *,
        reranker: BaseReranker | None = None,
        mmr_selector: DiversitySelector | None = None,
    ):
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker
        self.mmr_selector = mmr_selector

    @classmethod
    def from_components(
        cls,
        *,
        hybrid_searcher: HybridSearcher,
        reranker: BaseReranker | None = None,
        mmr_selector: MMRSelector | None = None,
    ) -> "RerankPipeline":
        return cls(hybrid_searcher=hybrid_searcher, reranker=reranker, mmr_selector=mmr_selector)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        options: RerankPipelineOptions | None = None,
    ) -> list[HybridSearchResult]:
        options = options or RerankPipelineOptions()
        options.validate()
        if options.top_k <= 0:
            return []

        hybrid_top_k = max(
            options.top_k,
            options.rerank_top_k if options.enable_reranker and self.reranker is not None else 0,
            options.mmr_pre_candidates if options.enable_mmr and self.mmr_selector is not None else 0,
        )
        candidates = self.hybrid_searcher.search(
            query,
            top_k=hybrid_top_k,
            dense_top_k=options.dense_top_k,
            sparse_top_k=options.sparse_top_k,
            filters=filters,
        )

        if options.enable_reranker and self.reranker is not None:
            candidates = self._safe_rerank(query, candidates)

        if options.enable_mmr and self.mmr_selector is not None:
            pre_candidates = candidates[: max(options.mmr_pre_candidates, options.top_k)]
            return self._safe_mmr(pre_candidates, top_k=min(options.mmr_top_k, options.top_k))

        return _rerank_final_slice(candidates, options.top_k)

    def search_dicts(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        options: RerankPipelineOptions | None = None,
    ) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.search(query, filters=filters, options=options)]

    def invoke(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        options: RerankPipelineOptions | None = None,
    ) -> list[str]:
        return [item.content for item in self.search(query, filters=filters, options=options)]

    def _safe_rerank(self, query: str, candidates: list[HybridSearchResult]) -> list[HybridSearchResult]:
        try:
            return self.reranker.rerank(query, candidates) if self.reranker is not None else candidates
        except Exception:
            return candidates

    def _safe_mmr(self, candidates: list[HybridSearchResult], *, top_k: int) -> list[HybridSearchResult]:
        try:
            return self.mmr_selector.select(candidates, top_k=top_k) if self.mmr_selector is not None else candidates[:top_k]
        except Exception:
            return candidates[:top_k]


def _rerank_final_slice(candidates: list[HybridSearchResult], top_k: int) -> list[HybridSearchResult]:
    selected = candidates[:top_k]
    for rank, candidate in enumerate(selected, 1):
        candidate.rank = rank
        if candidate.rerank_score is not None or candidate.mmr_score is not None:
            candidate.stage4_rank = rank
    return selected
