"""Common vector-store protocol for RAG V2 retrieval backends.

The protocol keeps Stage2 Qdrant code decoupled from later vector backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(slots=True)
class VectorPoint:
    """A vector and its retrievable payload.

    ``chunk_id`` is the stable business id used by the RAG pipeline. Concrete
    stores may map it to their own internal point id, but must keep it in
    payload for filtering and deletion.
    """

    chunk_id: str
    vector: Sequence[float]
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        try:
            vector_len = len(self.vector)
        except TypeError as exc:
            raise TypeError("vector must be a sized sequence") from exc
        if vector_len == 0:
            raise ValueError("vector must not be empty")
        self.payload = dict(self.payload)
        self.payload.setdefault("chunk_id", self.chunk_id)


@dataclass(slots=True)
class VectorSearchHit:
    """Backend-neutral vector search result."""

    rank: int
    chunk_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    point_id: str | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "payload": self.payload,
            "point_id": self.point_id,
        }


class VectorStore(Protocol):
    """Minimal operations required by Stage2 import, sync, and search."""

    def ensure_collection(self) -> None: ...

    def upsert_points(self, points: Sequence[VectorPoint]) -> None: ...

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None: ...

    def delete_by_source_file(self, source_file: str) -> None: ...

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]: ...

    def count(self) -> int: ...

    def health(self) -> dict[str, Any]: ...
