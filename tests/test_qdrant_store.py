from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag_v2.config import RagV2Config
from rag_v2.stores.qdrant_store import QdrantStore, QdrantStoreConfig
from rag_v2.stores.vector_store import VectorPoint


@pytest.fixture()
def qdrant_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "artifacts" / "stage2_test_qdrant"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    if path.exists():
        shutil.rmtree(path)


def make_store(path: Path, collection: str = "test_stage2_qdrant") -> QdrantStore:
    cfg = QdrantStoreConfig(
        mode="local",
        path=path,
        collection_name=collection,
        vector_size=3,
        distance="COSINE",
    )
    return QdrantStore(cfg)


def test_qdrant_config_resolves_from_rag_config():
    rag_cfg = RagV2Config.default()
    cfg = QdrantStoreConfig.from_rag_config(rag_cfg, vector_size=1024)
    assert cfg.mode == "local"
    assert cfg.path == rag_cfg.qdrant_path
    assert cfg.url is None
    assert cfg.collection_name == "youan_rag_stage2"
    assert cfg.vector_size == 1024


def test_qdrant_store_create_upsert_search_filter_and_delete(qdrant_path: Path):
    store = make_store(qdrant_path)
    try:
        assert store.count() == 0
        store.ensure_collection()
        assert store.health()["collection_exists"] is True

        store.upsert_points(
            [
                VectorPoint("chunk-1", [1.0, 0.0, 0.0], {"source_file": "a.md", "chunk_index": 0}),
                VectorPoint("chunk-2", [0.0, 1.0, 0.0], {"source_file": "a.md", "chunk_index": 1}),
                VectorPoint("chunk-3", [0.0, 0.0, 1.0], {"source_file": "b.md", "chunk_index": 0}),
            ]
        )
        assert store.count() == 3

        hits = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(hits) == 2
        assert hits[0].rank == 1
        assert hits[0].chunk_id == "chunk-1"
        assert hits[0].payload["source_file"] == "a.md"

        filtered = store.search([1.0, 0.0, 0.0], top_k=5, filters={"source_file": "b.md"})
        assert [hit.chunk_id for hit in filtered] == ["chunk-3"]

        store.delete_by_chunk_ids(["chunk-1"])
        assert store.count() == 2
        assert "chunk-1" not in [hit.chunk_id for hit in store.search([1.0, 0.0, 0.0], top_k=3)]

        store.delete_by_source_file("a.md")
        assert store.count() == 1
        remaining = store.search([0.0, 0.0, 1.0], top_k=3)
        assert [hit.chunk_id for hit in remaining] == ["chunk-3"]
    finally:
        store.close()


def test_qdrant_store_validates_vector_size(qdrant_path: Path):
    store = make_store(qdrant_path, collection="test_vector_size")
    try:
        with pytest.raises(ValueError, match="vector size mismatch"):
            store.upsert_points([VectorPoint("bad", [1.0, 0.0], {"source_file": "a.md"})])
        with pytest.raises(ValueError, match="query vector size mismatch"):
            store.search([1.0, 0.0], top_k=1)
    finally:
        store.close()
