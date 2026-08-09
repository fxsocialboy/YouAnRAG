"""Stage4 Cross-Encoder reranking utilities.

The module keeps the real reranker lazily loaded so unit tests and normal
imports never trigger a model download. Stage4 can therefore be developed and
validated with FakeReranker first, then switched to a real CrossEncoder in CLI
or evaluation scripts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from rag_v2.retrieval.hybrid_searcher import HybridSearchResult


class BaseReranker(Protocol):
    """Protocol for optional Stage4 rerankers."""

    def rerank(self, query: str, candidates: list[HybridSearchResult]) -> list[HybridSearchResult]: ...


class FakeReranker:
    """Deterministic reranker for tests and no-model pipeline smoke runs.

    Parameters
    ----------
    scores:
        Mapping from ``chunk_id`` to rerank score. Missing ids use
        ``default_score``.
    score_fn:
        Optional function ``(query, candidate) -> score``. If provided it wins
        over ``scores``.
    raise_on_rerank:
        Test hook. When true, rerank returns original candidates unchanged via
        the same fallback behavior used by the real reranker.
    """

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        *,
        score_fn: Callable[[str, HybridSearchResult], float] | None = None,
        default_score: float = 0.0,
        raise_on_rerank: bool = False,
    ):
        self.scores = scores or {}
        self.score_fn = score_fn
        self.default_score = default_score
        self.raise_on_rerank = raise_on_rerank

    def rerank(self, query: str, candidates: list[HybridSearchResult]) -> list[HybridSearchResult]:
        try:
            if self.raise_on_rerank:
                raise RuntimeError("fake reranker failure")
            return _apply_scores(
                candidates,
                [self._score(query, candidate) for candidate in candidates],
            )
        except Exception:
            return candidates

    def _score(self, query: str, candidate: HybridSearchResult) -> float:
        if self.score_fn is not None:
            return float(self.score_fn(query, candidate))
        return float(self.scores.get(candidate.chunk_id, self.default_score))


class CrossEncoderReranker:
    """Lazy wrapper around ``sentence_transformers.CrossEncoder``.

    It reranks a small candidate list by scoring ``(query, passage)`` pairs.
    Any model-loading or inference exception falls back to the original RRF
    order, so reranking never blocks the main retrieval path.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        model_factory: Callable[[], object] | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model_factory = model_factory
        self._model: object | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, candidates: list[HybridSearchResult]) -> list[HybridSearchResult]:
        if len(candidates) <= 1:
            return candidates
        try:
            pairs = [(query, candidate.content) for candidate in candidates]
            scores = self._predict(pairs)
            return _apply_scores(candidates, scores)
        except Exception:
            return candidates

    def close(self) -> None:
        self._model = None

    def _load_model(self) -> object:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory()
            else:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name_or_path,
                    device=self.device,
                    max_length=self.max_length,
                )
        return self._model

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load_model()
        predict = getattr(model, "predict")
        raw_scores = predict(pairs, batch_size=self.batch_size)
        return [float(score) for score in raw_scores]


def _apply_scores(candidates: list[HybridSearchResult], scores: Sequence[float]) -> list[HybridSearchResult]:
    if len(candidates) != len(scores):
        raise ValueError("scores length must match candidates length")
    if not candidates:
        return []

    for candidate, score in zip(candidates, scores, strict=True):
        candidate.rerank_score = round(float(score), 6)

    ranked = sorted(
        candidates,
        key=lambda item: (
            -(item.rerank_score if item.rerank_score is not None else float("-inf")),
            item.rank,
            item.chunk_id,
        ),
    )
    for rank, candidate in enumerate(ranked, 1):
        candidate.rank = rank
        candidate.stage4_rank = rank
    return ranked
