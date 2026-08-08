"""Stage-1 searcher over optimized chunks and FAISS index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from rag_v2.config import RagV2Config, get_config
from rag_v2.embedding.bge_embedder import BGEEmbedder, BGEEmbedderConfig
from rag_v2.stores.faiss_store import FaissStore


class QueryEmbedder(Protocol):
    def encode_queries(self, queries: list[str]) -> np.ndarray: ...


@dataclass(slots=True)
class Stage1SearchResult:
    rank: int
    global_index: int
    score_l2: float
    source_file: str
    chunk_index: int
    chunk_id: str
    section_path: list[str]
    section_path_text: str
    content: str
    content_preview: str
    token_count: int

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "global_index": self.global_index,
            "score_l2": self.score_l2,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "section_path": self.section_path,
            "section_path_text": self.section_path_text,
            "content": self.content,
            "content_preview": self.content_preview,
            "token_count": self.token_count,
        }


class Stage1Searcher:
    """Search optimized stage-1 FAISS index."""

    def __init__(self, store: FaissStore, embedder: QueryEmbedder):
        self.store = store
        self.embedder = embedder

    @classmethod
    def from_config(cls, cfg: RagV2Config | None = None, device: str = "cpu", batch_size: int = 16) -> "Stage1Searcher":
        cfg = cfg or get_config()
        cfg.validate(require_model=True)
        metadata_path = cfg.stage1_artifacts_dir / "chunk_metadata.json"
        index_path = cfg.stage1_artifacts_dir / "faiss_index.index"
        metadata = load_metadata(metadata_path)
        store = FaissStore.load(index_path, metadata=metadata)
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

    def search(self, query: str, top_k: int = 10) -> list[Stage1SearchResult]:
        query_vectors = self.embedder.encode_queries([query])
        hits = self.store.search(query_vectors, top_k=top_k)[0]
        results: list[Stage1SearchResult] = []
        for hit in hits:
            meta = hit.metadata or {}
            content = meta.get("content", "")
            results.append(
                Stage1SearchResult(
                    rank=hit.rank,
                    global_index=hit.index,
                    score_l2=hit.score_l2,
                    source_file=meta.get("source_file", ""),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    chunk_id=meta.get("chunk_id", ""),
                    section_path=list(meta.get("section_path", [])),
                    section_path_text=meta.get("section_path_text", ""),
                    content=content,
                    content_preview=content[:160],
                    token_count=int(meta.get("token_count", 0)),
                )
            )
        return results

    def search_dicts(self, query: str, top_k: int = 10) -> list[dict]:
        return [result.to_dict() for result in self.search(query, top_k=top_k)]


def load_metadata(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
