from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from rag_v2.retrieval.qdrant_searcher import QdrantSearcher
from rag_v2.stores.vector_store import VectorSearchHit

PROJECT_ROOT = Path(r"G:\tiaozhanbei\newrag")
EVAL_PATH = PROJECT_ROOT / "evaluate_stage2.py"

spec = importlib.util.spec_from_file_location("evaluate_stage2", EVAL_PATH)
evaluate_stage2_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(evaluate_stage2_module)


class FakeEmbedder:
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return np.asarray([[0.1, 0.2, 0.3] for _ in queries], dtype="float32")


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, query_vector, top_k: int, filters=None):
        self.calls.append({"query_vector": list(query_vector), "top_k": top_k, "filters": filters})
        return [
            VectorSearchHit(
                rank=1,
                chunk_id="doc-a.md::0",
                score=0.99,
                payload={
                    "source_file": "doc-a.md",
                    "chunk_index": 0,
                    "chunk_id": "doc-a.md::0",
                    "section_path": ["A"],
                    "section_path_text": "A",
                    "content": "alpha",
                    "token_count": 10,
                },
                point_id="point-0",
            )
        ]

    def close(self):
        return None


def test_qdrant_searcher_returns_structured_results_and_passes_filters():
    store = FakeStore()
    searcher = QdrantSearcher(store=store, embedder=FakeEmbedder())
    results = searcher.search("query", top_k=3, filters={"source_file": "doc-a.md"})
    assert len(results) == 1
    assert results[0].chunk_id == "doc-a.md::0"
    assert results[0].source_file == "doc-a.md"
    assert results[0].section_path_text == "A"
    assert results[0].content_preview == "alpha"
    assert store.calls[0]["filters"] == {"source_file": "doc-a.md"}


def test_qdrant_searcher_invoke_returns_content_list():
    searcher = QdrantSearcher(store=FakeStore(), embedder=FakeEmbedder())
    assert searcher.invoke("query", top_k=1) == ["alpha"]


def test_stage2_consistency_metric_helpers():
    prefix = evaluate_stage2_module.prefix_match_ratio(["a", "b", "c"], ["a", "x", "c"])
    overlap = evaluate_stage2_module.set_overlap_ratio(["a", "b", "c"], ["a", "x", "c"])
    payload = evaluate_stage2_module.payload_completeness(
        [
            {
                "chunk_id": "a",
                "source_file": "f",
                "chunk_index": 0,
                "section_path_text": "s",
                "content": "x",
                "content_hash": "h",
                "token_count": 1,
                "is_active": True,
            }
        ]
    )
    assert round(prefix, 4) == 0.6667
    assert round(overlap, 4) == 0.6667
    assert payload == 1.0
