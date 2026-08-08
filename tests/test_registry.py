from __future__ import annotations

from pathlib import Path

from rag_v2.sync.registry import ChunkMapping, DocumentRegistry


def test_registry_init_upsert_update_and_list(tmp_path: Path):
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    registry.init_schema()

    registry.upsert_document(
        source_file="a.md",
        relative_path="a.md",
        content_hash="hash-1",
        status="active",
        chunk_count=0,
    )
    doc = registry.get_document("a.md")
    assert doc is not None
    assert doc.content_hash == "hash-1"
    assert doc.status == "active"

    registry.upsert_document(
        source_file="a.md",
        relative_path="nested/a.md",
        content_hash="hash-2",
        status="active",
        chunk_count=3,
    )
    updated = registry.get_document("a.md")
    assert updated is not None
    assert updated.relative_path == "nested/a.md"
    assert updated.content_hash == "hash-2"
    assert updated.chunk_count == 3
    assert registry.document_hashes() == {"a.md": "hash-2"}


def test_registry_replace_chunk_mappings_and_mark_deleted(tmp_path: Path):
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    registry.upsert_document(
        source_file="a.md",
        relative_path="a.md",
        content_hash="doc-hash",
        chunk_count=0,
    )
    registry.replace_chunk_mappings(
        "a.md",
        [
            ChunkMapping("a.md::0", "a.md", 0, "chunk-hash-0", "point-0"),
            ChunkMapping("a.md::1", "a.md", 1, "chunk-hash-1", "point-1"),
        ],
    )
    mappings = registry.list_chunk_mappings("a.md")
    assert [m.chunk_id for m in mappings] == ["a.md::0", "a.md::1"]
    assert registry.get_document("a.md").chunk_count == 2

    registry.replace_chunk_mappings(
        "a.md",
        [ChunkMapping("a.md::2", "a.md", 0, "chunk-hash-2", "point-2")],
    )
    mappings = registry.list_chunk_mappings("a.md")
    assert [m.chunk_id for m in mappings] == ["a.md::2"]
    assert registry.get_document("a.md").chunk_count == 1

    registry.mark_deleted("a.md")
    deleted = registry.get_document("a.md")
    assert deleted.status == "deleted"
    assert deleted.chunk_count == 0
    assert registry.list_chunk_mappings("a.md") == []
    assert registry.document_hashes(active_only=True) == {}
    assert registry.document_hashes(active_only=False) == {"a.md": "doc-hash"}


def test_registry_rejects_cross_document_chunk_mapping(tmp_path: Path):
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    registry.upsert_document(source_file="a.md", relative_path="a.md", content_hash="hash")
    try:
        registry.replace_chunk_mappings("a.md", [ChunkMapping("b.md::0", "b.md", 0, "hash", "point")])
    except ValueError as exc:
        assert "same source_file" in str(exc)
    else:
        raise AssertionError("expected source_file validation error")
