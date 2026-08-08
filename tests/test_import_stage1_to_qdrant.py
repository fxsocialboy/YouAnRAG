from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import faiss
import numpy as np

from rag_v2.stores.qdrant_store import QdrantStore, QdrantStoreConfig

PROJECT_ROOT = Path(r"G:\tiaozhanbei\newrag")
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "import_stage1_to_qdrant.py"

spec = importlib.util.spec_from_file_location("import_stage1_to_qdrant", SCRIPT_PATH)
importer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(importer)


def make_metadata() -> list[dict]:
    return [
        {
            "chunk_id": "doc-a.md::0",
            "source_file": "doc-a.md",
            "chunk_index": 0,
            "section_path": ["A", "Intro"],
            "section_path_text": "A > Intro",
            "content": "alpha content",
            "content_hash": "hash-a0",
            "token_count": 10,
            "char_count": 13,
            "is_indexable": True,
        },
        {
            "chunk_id": "doc-a.md::1",
            "source_file": "doc-a.md",
            "chunk_index": 1,
            "section_path": ["A", "Plan"],
            "section_path_text": "A > Plan",
            "content": "beta content",
            "content_hash": "hash-a1",
            "token_count": 11,
            "char_count": 12,
            "is_indexable": True,
        },
        {
            "chunk_id": "doc-b.md::0",
            "source_file": "doc-b.md",
            "chunk_index": 0,
            "section_path": ["B"],
            "section_path_text": "B",
            "content": "gamma content",
            "content_hash": "hash-b0",
            "token_count": 12,
            "char_count": 13,
            "is_indexable": True,
        },
    ]


def make_faiss_index(path: Path) -> None:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    faiss.write_index(index, str(path))


def test_metadata_to_payload_contains_required_fields():
    payload = importer.metadata_to_payload(make_metadata()[0])
    assert payload["chunk_id"] == "doc-a.md::0"
    assert payload["source_file"] == "doc-a.md"
    assert payload["chunk_index"] == 0
    assert payload["section_path_text"] == "A > Intro"
    assert payload["content_hash"] == "hash-a0"
    assert payload["is_active"] is True


def test_import_stage1_to_qdrant_is_idempotent_and_searchable():
    base = PROJECT_ROOT / "artifacts" / "stage2_import_test"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    try:
        index_path = base / "faiss.index"
        metadata_path = base / "chunk_metadata.json"
        report_path = base / "qdrant_import_report.json"
        qdrant_path = base / "qdrant_local"
        make_faiss_index(index_path)
        metadata_path.write_text(json.dumps(make_metadata(), ensure_ascii=False), encoding="utf-8")

        cfg = QdrantStoreConfig(
            mode="local",
            path=qdrant_path,
            collection_name="test_import_stage1_to_qdrant",
            vector_size=3,
            distance="COSINE",
        )
        report = importer.import_stage1_to_qdrant(
            index_path=index_path,
            metadata_path=metadata_path,
            qdrant_config=cfg,
            report_path=report_path,
            batch_size=2,
            recreate=True,
        )
        assert report["imported_points"] == 3
        assert report["qdrant_count"] == 3
        assert report_path.exists()

        # Re-import without recreating the collection: point ids are derived from
        # chunk_id, so Qdrant upsert updates existing points instead of duplicating.
        second = importer.import_stage1_to_qdrant(
            index_path=index_path,
            metadata_path=metadata_path,
            qdrant_config=cfg,
            report_path=report_path,
            batch_size=2,
            recreate=False,
        )
        assert second["qdrant_count"] == 3

        store = QdrantStore(cfg)
        try:
            hits = store.search([1.0, 0.0, 0.0], top_k=1)
            assert hits[0].chunk_id == "doc-a.md::0"
            assert hits[0].payload["content"] == "alpha content"
            assert hits[0].payload["is_active"] is True
            filtered = store.search([1.0, 0.0, 0.0], top_k=5, filters={"source_file": "doc-b.md"})
            assert [hit.chunk_id for hit in filtered] == ["doc-b.md::0"]
        finally:
            store.close()
    finally:
        if base.exists():
            shutil.rmtree(base)


def test_import_stage1_rejects_metadata_count_mismatch():
    base = PROJECT_ROOT / "artifacts" / "stage2_import_mismatch_test"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    try:
        index_path = base / "faiss.index"
        metadata_path = base / "chunk_metadata.json"
        report_path = base / "qdrant_import_report.json"
        make_faiss_index(index_path)
        metadata_path.write_text(json.dumps(make_metadata()[:2], ensure_ascii=False), encoding="utf-8")
        cfg = QdrantStoreConfig(
            mode="local",
            path=base / "qdrant_local",
            collection_name="test_import_mismatch",
            vector_size=3,
        )
        try:
            importer.import_stage1_to_qdrant(
                index_path=index_path,
                metadata_path=metadata_path,
                qdrant_config=cfg,
                report_path=report_path,
            )
        except ValueError as exc:
            assert "metadata_count" in str(exc)
        else:
            raise AssertionError("expected metadata_count mismatch")
    finally:
        if base.exists():
            shutil.rmtree(base)
