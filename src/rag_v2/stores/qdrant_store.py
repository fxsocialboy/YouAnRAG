"""Qdrant-backed vector store for Stage2.

Stage2 defaults to Qdrant Local for a lightweight single-process project, while
keeping the URL mode configurable for a later Qdrant Server migration.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        FilterSelector,
        MatchAny,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent
    raise ImportError("qdrant-client is required for QdrantStore; install qdrant-client first") from exc

from rag_v2.config import RagV2Config
from rag_v2.stores.vector_store import VectorPoint, VectorSearchHit


_QDRANT_ID_NAMESPACE = uuid.UUID("4da9b92d-4d2a-4b91-92bb-d5c9f15e1d25")


@dataclass(frozen=True, slots=True)
class QdrantStoreConfig:
    """Resolved settings for QdrantStore."""

    mode: str = "local"
    path: Path | None = None
    url: str | None = None
    collection_name: str = "youan_rag_stage2"
    vector_size: int = 1024
    distance: str = "COSINE"

    @classmethod
    def from_rag_config(cls, cfg: RagV2Config, vector_size: int = 1024) -> "QdrantStoreConfig":
        return cls(
            mode=cfg.qdrant_mode,
            path=cfg.qdrant_path,
            url=cfg.qdrant_url,
            collection_name=cfg.qdrant_collection,
            vector_size=vector_size,
        )


class QdrantStore:
    """Small Qdrant wrapper implementing the Stage2 vector-store contract."""

    def __init__(self, config: QdrantStoreConfig):
        self.config = config
        self.client = self._create_client(config)

    @staticmethod
    def _create_client(config: QdrantStoreConfig) -> QdrantClient:
        if config.mode == "local":
            if config.path is None:
                raise ValueError("qdrant local mode requires path")
            config.path.mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=str(config.path))
        if config.mode == "server":
            if not config.url:
                raise ValueError("qdrant server mode requires url")
            return QdrantClient(url=config.url)
        raise ValueError(f"unsupported qdrant mode: {config.mode}")

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            actual_size = _extract_vector_size(info)
            if actual_size is not None and actual_size != self.config.vector_size:
                raise ValueError(
                    f"collection vector size mismatch: {actual_size} != {self.config.vector_size}"
                )
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.config.vector_size, distance=_distance(self.config.distance)),
        )

    def upsert_points(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        self.ensure_collection()
        qdrant_points: list[PointStruct] = []
        for point in points:
            vector = [float(value) for value in point.vector]
            if len(vector) != self.config.vector_size:
                raise ValueError(f"vector size mismatch for {point.chunk_id}: {len(vector)} != {self.config.vector_size}")
            payload = dict(point.payload)
            payload["chunk_id"] = point.chunk_id
            qdrant_points.append(
                PointStruct(id=chunk_id_to_qdrant_point_id(point.chunk_id), vector=vector, payload=payload)
            )
        self.client.upsert(collection_name=self.collection_name, points=qdrant_points)

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids or not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=_field_filter("chunk_id", chunk_ids)),
        )

    def delete_by_source_file(self, source_file: str) -> None:
        if not source_file or not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=_field_filter("source_file", source_file)),
        )

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]:
        if top_k <= 0 or not self.client.collection_exists(self.collection_name):
            return []
        vector = [float(value) for value in query_vector]
        if len(vector) != self.config.vector_size:
            raise ValueError(f"query vector size mismatch: {len(vector)} != {self.config.vector_size}")
        query_filter = _payload_filter(filters or {})
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        points = getattr(response, "points", response)
        hits: list[VectorSearchHit] = []
        for rank, scored in enumerate(points, 1):
            payload = dict(scored.payload or {})
            chunk_id = str(payload.get("chunk_id", scored.id))
            hits.append(
                VectorSearchHit(
                    rank=rank,
                    chunk_id=chunk_id,
                    score=float(scored.score),
                    payload=payload,
                    point_id=scored.id,
                )
            )
        return hits

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)

    def health(self) -> dict[str, Any]:
        exists = self.client.collection_exists(self.collection_name)
        return {
            "backend": "qdrant",
            "mode": self.config.mode,
            "collection": self.collection_name,
            "collection_exists": exists,
            "vector_size": self.config.vector_size,
            "count": self.count() if exists else 0,
            "status": "ok",
        }


def chunk_id_to_qdrant_point_id(chunk_id: str) -> str:
    """Map a business chunk_id to a stable Qdrant-compatible UUID string."""

    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, chunk_id))


def _distance(name: str) -> Distance:
    normalized = name.upper()
    if normalized == "COSINE":
        return Distance.COSINE
    if normalized in {"EUCLID", "L2"}:
        return Distance.EUCLID
    if normalized == "DOT":
        return Distance.DOT
    raise ValueError(f"unsupported qdrant distance: {name}")


def _field_filter(key: str, value: Any) -> Filter:
    if isinstance(value, list):
        return Filter(must=[FieldCondition(key=key, match=MatchAny(any=[str(item) for item in value]))])
    return Filter(must=[FieldCondition(key=key, match=MatchValue(value=str(value)))])


def _payload_filter(filters: dict[str, Any]) -> Filter | None:
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, list):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=[str(item) for item in value])))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=str(value))))
    return Filter(must=conditions) if conditions else None


def _extract_vector_size(collection_info: Any) -> int | None:
    """Best-effort vector-size extraction across qdrant-client versions."""

    vectors = getattr(getattr(collection_info, "config", None), "params", None)
    vectors = getattr(vectors, "vectors", None)
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, dict) and vectors:
        first = next(iter(vectors.values()))
        if hasattr(first, "size"):
            return int(first.size)
    return None
