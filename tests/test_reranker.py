from __future__ import annotations

import pytest

from rag_v2.retrieval.hybrid_searcher import HybridSearchResult
from rag_v2.retrieval.reranker import CrossEncoderReranker, FakeReranker


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


def test_fake_reranker_sorts_by_score_and_updates_rank_fields():
    candidates = [make_candidate("a", 1), make_candidate("b", 2), make_candidate("c", 3)]
    reranker = FakeReranker({"a": 0.2, "b": 0.9, "c": 0.5})

    results = reranker.rerank("query", candidates)

    assert [item.chunk_id for item in results] == ["b", "c", "a"]
    assert [item.rank for item in results] == [1, 2, 3]
    assert [item.stage4_rank for item in results] == [1, 2, 3]
    assert [item.rerank_score for item in results] == [0.9, 0.5, 0.2]
    assert results[0].to_dict()["rerank_score"] == 0.9


def test_fake_reranker_keeps_stable_order_for_tied_scores():
    candidates = [make_candidate("b", 2), make_candidate("a", 1), make_candidate("c", 3)]
    reranker = FakeReranker({"a": 0.5, "b": 0.5, "c": 0.5})

    results = reranker.rerank("query", candidates)

    assert [item.chunk_id for item in results] == ["a", "b", "c"]


def test_fake_reranker_handles_empty_and_single_candidate():
    reranker = FakeReranker({"a": 1.0})
    assert reranker.rerank("query", []) == []

    candidate = make_candidate("a", 1)
    results = reranker.rerank("query", [candidate])
    assert results == [candidate]
    assert results[0].rerank_score == 1.0


def test_fake_reranker_exception_falls_back_to_original_candidates_unchanged():
    candidates = [make_candidate("a", 1), make_candidate("b", 2)]
    reranker = FakeReranker({"b": 1.0}, raise_on_rerank=True)

    results = reranker.rerank("query", candidates)

    assert results is candidates
    assert [item.chunk_id for item in results] == ["a", "b"]
    assert all(item.rerank_score is None for item in results)
    assert all(item.stage4_rank is None for item in results)


class DummyCrossEncoder:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, batch_size=16):
        self.calls.append({"pairs": pairs, "batch_size": batch_size})
        return [0.1, 0.8, 0.4]


def test_cross_encoder_reranker_is_lazy_and_uses_injected_model():
    dummy = DummyCrossEncoder()
    reranker = CrossEncoderReranker(
        "dummy-model",
        device="cpu",
        batch_size=4,
        max_length=128,
        model_factory=lambda: dummy,
    )
    candidates = [make_candidate("a", 1), make_candidate("b", 2), make_candidate("c", 3)]

    assert reranker.model_loaded is False
    results = reranker.rerank("query", candidates)

    assert reranker.model_loaded is True
    assert dummy.calls[0]["batch_size"] == 4
    assert dummy.calls[0]["pairs"][0] == ("query", "a content")
    assert [item.chunk_id for item in results] == ["b", "c", "a"]


def test_cross_encoder_reranker_falls_back_on_model_exception():
    class FailingModel:
        def predict(self, pairs, batch_size=16):
            raise RuntimeError("boom")

    candidates = [make_candidate("a", 1), make_candidate("b", 2)]
    reranker = CrossEncoderReranker("dummy-model", model_factory=lambda: FailingModel())

    results = reranker.rerank("query", candidates)

    assert results is candidates
    assert [item.chunk_id for item in results] == ["a", "b"]
    assert all(item.rerank_score is None for item in results)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"max_length": 0}])
def test_cross_encoder_reranker_validates_positive_runtime_params(kwargs):
    with pytest.raises(ValueError):
        CrossEncoderReranker("dummy-model", **kwargs)
