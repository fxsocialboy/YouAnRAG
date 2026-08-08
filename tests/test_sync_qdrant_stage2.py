from __future__ import annotations

import importlib.util
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from rag_v2.config import RagV2Config
from rag_v2.stores.qdrant_store import QdrantStore, QdrantStoreConfig
from rag_v2.sync.registry import DocumentRegistry

PROJECT_ROOT = Path(r"G:\tiaozhanbei\newrag")
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sync_qdrant_stage2.py"

spec = importlib.util.spec_from_file_location("sync_qdrant_stage2", SCRIPT_PATH)
sync_script = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sync_script)


class FakeEmbedder:
    def encode_passages(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = (sum(ord(ch) for ch in text) % 997) / 997.0
            vectors.append([1.0, seed, 0.5])
        return np.asarray(vectors, dtype="float32")


def prepare_env(name: str):
    base = PROJECT_ROOT / "artifacts" / name
    if base.exists():
        shutil.rmtree(base)
    markdown_dir = base / "final_mds"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(
        RagV2Config.default(),
        source_markdown_dir=markdown_dir,
        stage2_artifacts_dir=base / "stage2",
        qdrant_path=base / "qdrant_local",
        registry_db_path=base / "stage2" / "document_registry.sqlite3",
    )
    qdrant_cfg = QdrantStoreConfig(
        mode="local",
        path=cfg.qdrant_path,
        collection_name=f"{name}_collection",
        vector_size=3,
    )
    registry = DocumentRegistry(cfg.registry_db_path)
    store = QdrantStore(qdrant_cfg)
    return base, markdown_dir, cfg, registry, store


def test_scan_only_detects_added_without_writing_registry_or_qdrant():
    base, markdown_dir, cfg, registry, store = prepare_env("stage2_sync_scan_test")
    try:
        (markdown_dir / "test_sync.md").write_text("# temp test\nnew content", encoding="utf-8")
        report = sync_script.scan_only_report(cfg, registry, source_file="test_sync.md")
        assert report["changes"]["added"] == 1
        assert report["added"][0]["source_file"] == "test_sync.md"
        assert not cfg.registry_db_path.exists()
        assert store.count() == 0
    finally:
        store.close()
        if base.exists():
            shutil.rmtree(base)


def test_sync_added_modified_deleted_and_idempotent():
    base, markdown_dir, cfg, registry, store = prepare_env("stage2_sync_lifecycle_test")
    try:
        path = markdown_dir / "test_sync.md"
        path.write_text("# temp test\nversion one content for add sync.", encoding="utf-8")

        first = sync_script.apply_sync(
            cfg=cfg,
            registry=registry,
            store=store,
            embedder=FakeEmbedder(),
            source_file="test_sync.md",
        )
        assert first["changes"]["added"] == 1
        assert first["qdrant_count"] >= 1
        count_after_add = store.count()
        hits = store.search([1.0, 0.0, 0.5], top_k=5, filters={"source_file": "test_sync.md"})
        assert hits
        assert "version one content" in hits[0].payload["content"]
        assert registry.get_document("test_sync.md").status == "active"
        assert len(registry.list_chunk_mappings("test_sync.md")) == count_after_add

        second = sync_script.apply_sync(
            cfg=cfg,
            registry=registry,
            store=store,
            embedder=FakeEmbedder(),
            source_file="test_sync.md",
        )
        assert second["changes"]["unchanged"] == 1
        assert store.count() == count_after_add

        path.write_text("# temp test\nversion two content for modified sync.", encoding="utf-8")
        modified = sync_script.apply_sync(
            cfg=cfg,
            registry=registry,
            store=store,
            embedder=FakeEmbedder(),
            source_file="test_sync.md",
        )
        assert modified["changes"]["modified"] == 1
        hits = store.search([1.0, 0.0, 0.5], top_k=5, filters={"source_file": "test_sync.md"})
        assert hits
        contents = "\n".join(hit.payload["content"] for hit in hits)
        assert "version two content" in contents
        assert "version one content" not in contents

        path.unlink()
        deleted = sync_script.apply_sync(
            cfg=cfg,
            registry=registry,
            store=store,
            embedder=FakeEmbedder(),
            source_file="test_sync.md",
        )
        assert deleted["changes"]["deleted"] == 1
        assert store.search([1.0, 0.0, 0.5], top_k=5, filters={"source_file": "test_sync.md"}) == []
        assert registry.get_document("test_sync.md").status == "deleted"
    finally:
        store.close()
        if base.exists():
            shutil.rmtree(base)
