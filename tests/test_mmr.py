from __future__ import annotations

import numpy as np
import pytest

from rag_v2.retrieval.hybrid_searcher import HybridSearchResult
from rag_v2.retrieval.mmr import MMRSelector, candidate_relevance, cosine_similarity_matrix, normalize_scores


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls = []

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        self.calls.append(passages)
        return np.asarray([self.vectors[text] for text in passages], dtype=np.float32)


class FailingEmbedder:
    def encode_passages(self, passages: list[str]) -> np.ndarray:
        raise RuntimeError("embedding failed")


def make_candidate(
    chunk_id: str,
    *,
    rank: int,
    fusion_score: float = 0.1,
    rerank_score: float | None = None,
    content: str | None = None,
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        content=content or chunk_id,
        source_file="doc.md",
        chunk_index=rank,
        section_path_text="Section",
        dense_score=None,
        sparse_score=None,
        fusion_score=fusion_score,
        rank=rank,
        metadata={},
        rerank_score=rerank_score,
    )


def test_normalize_scores_min_max_and_equal_values():
    assert np.allclose(normalize_scores([2.0, 4.0, 6.0]), np.asarray([0.0, 0.5, 1.0], dtype=np.float32))
    assert np.allclose(normalize_scores([3.0, 3.0]), np.asarray([1.0, 1.0], dtype=np.float32))


def test_cosine_similarity_matrix_normalizes_vectors_and_handles_zero():
    vectors = np.asarray([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    sims = cosine_similarity_matrix(vectors)
    assert sims.shape == (3, 3)
    assert sims[0, 1] == pytest.approx(1.0)
    assert sims[0, 2] == pytest.approx(0.0)


def test_candidate_relevance_prefers_rerank_score_over_fusion_score():
    candidate = make_candidate("a", rank=1, fusion_score=0.2, rerank_score=0.9)
    assert candidate_relevance(candidate) == 0.9
    candidate.rerank_score = None
    assert candidate_relevance(candidate) == 0.2


def test_mmr_penalizes_similar_chunk_and_selects_diverse_candidate():
    candidates = [
        make_candidate("a", rank=1, rerank_score=1.0),
        make_candidate("b", rank=2, rerank_score=0.9),
        make_candidate("c", rank=3, rerank_score=0.8),
    ]
    embedder = FakeEmbedder(
        {
            "a": [1.0, 0.0],
            "b": [1.0, 0.0],
            "c": [0.0, 1.0],
        }
    )
    selector = MMRSelector(embedder, lambda_=0.5)

    selected = selector.select(candidates, top_k=2)

    assert [item.chunk_id for item in selected] == ["a", "c"]
    assert selected[0].mmr_score == pytest.approx(1.0)
    assert selected[1].mmr_score is not None
    assert [item.rank for item in selected] == [1, 2]
    assert [item.stage4_rank for item in selected] == [1, 2]


def test_mmr_lambda_one_degenerates_to_relevance_order():
    candidates = [
        make_candidate("a", rank=1, rerank_score=0.7),
        make_candidate("b", rank=2, rerank_score=0.9),
        make_candidate("c", rank=3, rerank_score=0.8),
    ]
    embedder = FakeEmbedder({"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]})
    selector = MMRSelector(embedder, lambda_=1.0)

    selected = selector.select(candidates, top_k=3)

    assert [item.chunk_id for item in selected] == ["b", "c", "a"]


def test_mmr_lambda_zero_prefers_diversity_after_first_relevance_pick():
    candidates = [
        make_candidate("a", rank=1, rerank_score=1.0),
        make_candidate("b", rank=2, rerank_score=0.9),
        make_candidate("c", rank=3, rerank_score=0.1),
    ]
    embedder = FakeEmbedder({"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]})
    selector = MMRSelector(embedder, lambda_=0.0)

    selected = selector.select(candidates, top_k=2)

    assert [item.chunk_id for item in selected] == ["a", "c"]


def test_mmr_top_k_boundaries_and_single_candidate():
    candidate = make_candidate("a", rank=5, rerank_score=0.8)
    selector = MMRSelector(FakeEmbedder({"a": [1.0, 0.0]}))

    assert selector.select([candidate], top_k=0) == []
    selected = selector.select([candidate], top_k=10)
    assert selected == [candidate]
    assert selected[0].rank == 1
    assert selected[0].stage4_rank == 1
    assert selected[0].mmr_score == 1.0


def test_mmr_falls_back_to_original_slice_on_embedding_exception():
    candidates = [make_candidate("a", rank=1, rerank_score=0.8), make_candidate("b", rank=2, rerank_score=0.9)]
    selector = MMRSelector(FailingEmbedder())

    selected = selector.select(candidates, top_k=1)

    assert selected == candidates[:1]
    assert selected[0].chunk_id == "a"
    assert all(item.mmr_score is None for item in candidates)


@pytest.mark.parametrize("lambda_", [-0.1, 1.1])
def test_mmr_validates_lambda_range(lambda_: float):
    with pytest.raises(ValueError):
        MMRSelector(FakeEmbedder({}), lambda_=lambda_)
