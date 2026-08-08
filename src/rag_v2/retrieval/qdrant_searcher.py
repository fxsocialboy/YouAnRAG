"""Stage2 searcher over Qdrant Local/Server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rag_v2.config import RagV2Config, get_config
from rag_v2.embedding.bge_embedder import BGEEmbedder, BGEEmbedderConfig
from rag_v2.stores.qdrant_store import QdrantStore, QdrantStoreConfig


class QueryEmbedder(Protocol):
    def encode_queries(self, queries: list[str]) -> np.ndarray: ...


@dataclass(slots=True)
class Stage2SearchResult:
    rank: int
    score: float
    source_file: str
    chunk_index: int
    chunk_id: str
    section_path: list[str]
    section_path_text: str
    content: str
    content_preview: str
    token_count: int
    content_hash: str = ""
    is_active: bool = True
    point_id: str | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "section_path": self.section_path,
            "section_path_text": self.section_path_text,
            "content": self.content,
            "content_preview": self.content_preview,
            "token_count": self.token_count,
            "content_hash": self.content_hash,
            "is_active": self.is_active,
            "point_id": self.point_id,
        }


class QdrantSearcher:
    def __init__(self, store: QdrantStore, embedder: QueryEmbedder):
        self.store = store
        self.embedder = embedder

    @classmethod
    def from_config(cls, cfg: RagV2Config | None = None, device: str = "cpu", batch_size: int = 16) -> "QdrantSearcher":
        cfg = cfg or get_config()
        cfg.validate(require_model=True)
        store = QdrantStore(QdrantStoreConfig.from_rag_config(cfg, vector_size=1024))
        embedder = BGEEmbedder(
            BGEEmbedderConfig(
                model_path=cfg.model_path,
                device=device,
                dtype="float32",
                batch_size=batch_size,
                max_length=512,
                use_query_instruction=cfg.use_query_instruction,
                query_instruction=cfg.query_instruction,
            )
        )
        return cls(store=store, embedder=embedder)

    def close(self) -> None:
        close = getattr(self.store, 'close', None)
        if callable(close):
            close()

    def search(self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None) -> list[Stage2SearchResult]:
        query_vector = self.embedder.encode_queries([query])[0]
        hits = self.store.search(query_vector, top_k=top_k, filters=filters)
        results: list[Stage2SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            content = str(payload.get("content", ""))
            results.append(
                Stage2SearchResult(
                    rank=hit.rank,
                    score=float(hit.score),
                    source_file=str(payload.get("source_file", "")),
                    chunk_index=int(payload.get("chunk_index", -1)),
                    chunk_id=str(payload.get("chunk_id", hit.chunk_id)),
                    section_path=list(payload.get("section_path", [])),
                    section_path_text=str(payload.get("section_path_text", "")),
                    content=content,
                    content_preview=content[:160],
                    token_count=int(payload.get("token_count", 0)),
                    content_hash=str(payload.get("content_hash", "")),
                    is_active=bool(payload.get("is_active", True)),
                    point_id=hit.point_id,
                )
            )
        return results

    def search_dicts(self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.search(query, top_k=top_k, filters=filters)]

    def invoke(self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None) -> list[str]:
        return [item.content for item in self.search(query, top_k=top_k, filters=filters)]
