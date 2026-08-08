"""FAISS store helpers for stage 1.6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


@dataclass(slots=True)
class FaissSearchHit:
    rank: int
    index: int
    score_l2: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "index": self.index,
            "score_l2": self.score_l2,
            "metadata": self.metadata,
        }


class FaissStore:
    """Thin wrapper around ``faiss.IndexFlatL2`` with input validation."""

    def __init__(self, index: faiss.Index | None = None, metadata: list[dict[str, Any]] | None = None):
        self.index = index
        self.metadata = metadata or []

    @classmethod
    def from_vectors(cls, vectors: np.ndarray, metadata: list[dict[str, Any]] | None = None) -> "FaissStore":
        vectors = _ensure_float32_2d(vectors)
        if metadata is not None and len(metadata) != len(vectors):
            raise ValueError("metadata length must equal vector count")
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, metadata=metadata or [])

    @classmethod
    def load(cls, index_path: str | Path, metadata: list[dict[str, Any]] | None = None) -> "FaissStore":
        index = faiss.read_index(str(index_path))
        store = cls(index=index, metadata=metadata or [])
        store.validate()
        return store

    @property
    def ntotal(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    @property
    def dim(self) -> int:
        return 0 if self.index is None else int(self.index.d)

    def save(self, index_path: str | Path) -> None:
        if self.index is None:
            raise ValueError("index is not initialized")
        out = Path(index_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out))

    def validate(self) -> None:
        if self.index is None:
            raise ValueError("index is not initialized")
        if self.metadata and self.index.ntotal != len(self.metadata):
            raise ValueError(f"index.ntotal={self.index.ntotal} != metadata size={len(self.metadata)}")

    def search(self, query_vectors: np.ndarray, top_k: int = 10) -> list[list[FaissSearchHit]]:
        if self.index is None:
            raise ValueError("index is not initialized")
        self.validate()
        query_vectors = _ensure_float32_2d(query_vectors)
        if query_vectors.shape[1] != self.index.d:
            raise ValueError(f"query dim={query_vectors.shape[1]} != index dim={self.index.d}")
        safe_top_k = min(max(0, top_k), self.index.ntotal)
        if safe_top_k == 0:
            return [[] for _ in range(len(query_vectors))]
        distances, indices = self.index.search(query_vectors, safe_top_k)
        results: list[list[FaissSearchHit]] = []
        for row_distances, row_indices in zip(distances, indices):
            hits: list[FaissSearchHit] = []
            for rank, (idx, score) in enumerate(zip(row_indices.tolist(), row_distances.tolist()), 1):
                if idx < 0:
                    continue
                meta = self.metadata[idx] if self.metadata and idx < len(self.metadata) else None
                hits.append(FaissSearchHit(rank=rank, index=int(idx), score_l2=float(score), metadata=meta))
            results.append(hits)
        return results


def _ensure_float32_2d(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if vectors.shape[0] > 0 and vectors.shape[1] <= 0:
        raise ValueError("vector dimension must be positive")
    return vectors
