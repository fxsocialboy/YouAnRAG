from __future__ import annotations

import pytest

from rag_v2.retrieval.hybrid_searcher import HybridSearchResult
from rag_v2.retrieval.rerank_pipeline import RerankPipeline, RerankPipelineOptions
from rag_v2.retrieval.reranker import FakeReranker


def make_candidate(chunk_id: str, rank: int, fusion_score: float = 0.01) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        content=f"{chunk_id} content",
        source_file="doc.md",
        chunk_index=rank,
        section_path_text="Section",
        dense_score=None,
        sparse_score=None,
        fusion_score=fusion_score,
        rank=rank,
        metadata={},
    )


class FakeHybridSearcher:
    def __init__(self, candidates: list[HybridSearchResult]):
        self.candidates = candidates
        self.calls = []

    def search(self, query, *, top_k=10, dense_top_k=30, sparse_top_k=30, filters=None):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "dense_top_k": dense_top_k,
                "sparse_top_k": sparse_top_k,
                "filters": filters,
            }
        )
        return self.candidates[:top_k]


class FakeMMRSelector:
    def __init__(self, order: list[str] | None = None, raise_on_select: bool = False):
        self.order = order or []
        self.raise_on_select = raise_on_select
        self.calls = []

    def select(self, candidates: list[HybridSearchResult], top_k: int):
        self.calls.append({"candidates": list(candidates), "top_k": top_k})
        if self.raise_on_select:
            raise RuntimeError("mmr failed")
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        ordered = [by_id[chunk_id] for chunk_id in self.order if chunk_id in by_id]
        ordered.extend([candidate for candidate in candidates if candidate.chunk_id not in self.order])
        selected = ordered[:top_k]
        for rank, candidate in enumerate(selected, 1):
            candidate.mmr_score = round(1.0 / rank, 6)
            candidate.rank = rank
            candidate.stage4_rank = rank
        return selected


def test_pipeline_plain_hybrid_when_rerank_and_mmr_disabled():
    hybrid = FakeHybridSearcher([make_candidate("a", 1, 0.3), make_candidate("b", 2, 0.2)])
    pipeline = RerankPipeline(hybrid)
    options = RerankPipelineOptions(top_k=1, enable_reranker=False, enable_mmr=False)

    results = pipeline.search("query", filters={"source_file": "doc.md"}, options=options)

    assert [item.chunk_id for item in results] == ["a"]
    assert hybrid.calls[0]["top_k"] == 1
    assert hybrid.calls[0]["filters"] == {"source_file": "doc.md"}
    assert results[0].rerank_score is None
    assert results[0].mmr_score is None


def test_pipeline_applies_reranker_and_slices_top_k():
    hybrid = FakeHybridSearcher([make_candidate("a", 1), make_candidate("b", 2), make_candidate("c", 3)])
    reranker = FakeReranker({"a": 0.1, "b": 0.9, "c": 0.5})
    pipeline = RerankPipeline(hybrid, reranker=reranker)
    options = RerankPipelineOptions(top_k=2, rerank_top_k=3, enable_reranker=True, enable_mmr=False)

    results = pipeline.search("query", options=options)

    assert hybrid.calls[0]["top_k"] == 3
    assert [item.chunk_id for item in results] == ["b", "c"]
    assert [item.rerank_score for item in results] == [0.9, 0.5]
    assert [item.stage4_rank for item in results] == [1, 2]


def test_pipeline_applies_reranker_then_mmr():
    hybrid = FakeHybridSearcher([make_candidate("a", 1), make_candidate("b", 2), make_candidate("c", 3)])
    reranker = FakeReranker({"a": 0.1, "b": 0.9, "c": 0.5})
    mmr = FakeMMRSelector(order=["c", "b", "a"])
    pipeline = RerankPipeline(hybrid, reranker=reranker, mmr_selector=mmr)
    options = RerankPipelineOptions(
        top_k=2,
        rerank_top_k=3,
        mmr_pre_candidates=3,
        mmr_top_k=2,
        enable_reranker=True,
        enable_mmr=True,
    )

    results = pipeline.search("query", options=options)

    assert [item.chunk_id for item in results] == ["c", "b"]
    assert mmr.calls[0]["top_k"] == 2
    assert [item.mmr_score for item in results] == [1.0, 0.5]


def test_pipeline_reranker_exception_degrades_to_hybrid_order():
    hybrid = FakeHybridSearcher([make_candidate("a", 1), make_candidate("b", 2)])
    reranker = FakeReranker({"b": 1.0}, raise_on_rerank=True)
    pipeline = RerankPipeline(hybrid, reranker=reranker)
    options = RerankPipelineOptions(top_k=2, rerank_top_k=2, enable_reranker=True, enable_mmr=False)

    results = pipeline.search("query", options=options)

    assert [item.chunk_id for item in results] == ["a", "b"]
    assert all(item.rerank_score is None for item in results)


def test_pipeline_mmr_exception_degrades_to_pre_mmr_slice():
    hybrid = FakeHybridSearcher([make_candidate("a", 1), make_candidate("b", 2)])
    mmr = FakeMMRSelector(raise_on_select=True)
    pipeline = RerankPipeline(hybrid, mmr_selector=mmr)
    options = RerankPipelineOptions(top_k=1, mmr_pre_candidates=2, mmr_top_k=1, enable_reranker=False, enable_mmr=True)

    results = pipeline.search("query", options=options)

    assert [item.chunk_id for item in results] == ["a"]
    assert results[0].mmr_score is None


def test_pipeline_search_dicts_and_invoke_outputs():
    hybrid = FakeHybridSearcher([make_candidate("a", 1), make_candidate("b", 2)])
    pipeline = RerankPipeline(hybrid)
    options = RerankPipelineOptions(top_k=2, enable_reranker=False, enable_mmr=False)

    dicts = pipeline.search_dicts("query", options=options)
    texts = pipeline.invoke("query", options=options)

    assert dicts[0]["chunk_id"] == "a"
    assert texts == ["a content", "b content"]


@pytest.mark.parametrize("field", ["top_k", "dense_top_k", "sparse_top_k", "rerank_top_k", "mmr_pre_candidates", "mmr_top_k"])
def test_pipeline_options_validate_non_negative_fields(field: str):
    kwargs = {field: -1}
    options = RerankPipelineOptions(**kwargs)
    with pytest.raises(ValueError):
        options.validate()
