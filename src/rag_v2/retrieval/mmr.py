"""Stage4 MMR diversity selection.

MMR is applied after hybrid recall and optional rerank. It keeps highly relevant
chunks while penalizing candidates that are too similar to already selected
chunks, reducing near-duplicate evidence in the final Top-K.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from rag_v2.retrieval.hybrid_searcher import HybridSearchResult


class PassageEmbedder(Protocol):
    def encode_passages(self, passages: list[str]) -> np.ndarray: ...


class MMRSelector:
    """Select a diverse candidate subset with Maximal Marginal Relevance."""

    def __init__(self, embedder: PassageEmbedder, *, lambda_: float = 0.7):
        if not (0.0 <= lambda_ <= 1.0):
            raise ValueError("lambda_ must be between 0 and 1")
        self.embedder = embedder
        self.lambda_ = lambda_

    def select(self, candidates: list[HybridSearchResult], top_k: int) -> list[HybridSearchResult]:
        if top_k <= 0 or not candidates:
            return []
        if len(candidates) == 1:
            candidates[0].mmr_score = 1.0
            candidates[0].rank = 1
            candidates[0].stage4_rank = 1
            return candidates[:1]

        limit = min(top_k, len(candidates))
        try:
            relevance = normalize_scores([candidate_relevance(candidate) for candidate in candidates])
            embeddings = np.asarray(self.embedder.encode_passages([candidate.content for candidate in candidates]), dtype=np.float32)
            similarities = cosine_similarity_matrix(embeddings)
            selected_indexes = self._greedy_select(relevance, similarities, limit)
            selected = [candidates[idx] for idx in selected_indexes]
            for rank, (candidate, idx) in enumerate(zip(selected, selected_indexes, strict=True), 1):
                candidate.mmr_score = round(float(self._score_index(idx, selected_indexes[: rank - 1], relevance, similarities)), 6)
                candidate.rank = rank
                candidate.stage4_rank = rank
            return selected
        except Exception:
            return candidates[:limit]

    def _greedy_select(self, relevance: np.ndarray, similarities: np.ndarray, limit: int) -> list[int]:
        remaining = set(range(len(relevance)))
        selected: list[int] = []

        first = max(remaining, key=lambda idx: (float(relevance[idx]), -idx))
        selected.append(first)
        remaining.remove(first)

        while remaining and len(selected) < limit:
            next_idx = max(
                remaining,
                key=lambda idx: (self._score_index(idx, selected, relevance, similarities), float(relevance[idx]), -idx),
            )
            selected.append(next_idx)
            remaining.remove(next_idx)
        return selected

    def _score_index(self, idx: int, selected: list[int], relevance: np.ndarray, similarities: np.ndarray) -> float:
        if not selected:
            return float(relevance[idx])
        max_sim = max(float(similarities[idx, selected_idx]) for selected_idx in selected)
        return self.lambda_ * float(relevance[idx]) - (1.0 - self.lambda_) * max_sim


def candidate_relevance(candidate: HybridSearchResult) -> float:
    if candidate.rerank_score is not None:
        return float(candidate.rerank_score)
    return float(candidate.fusion_score)


def normalize_scores(scores: list[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        return values
    min_score = float(values.min())
    max_score = float(values.max())
    if max_score == min_score:
        return np.ones_like(values, dtype=np.float32)
    return (values - min_score) / (max_score - min_score)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if embeddings.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return np.clip(normalized @ normalized.T, -1.0, 1.0).astype(np.float32)
