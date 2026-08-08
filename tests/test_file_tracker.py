from __future__ import annotations

from pathlib import Path

from rag_v2.sync.file_tracker import detect_changes, scan_markdown_files
from rag_v2.sync.registry import DocumentRegistry


def test_scan_markdown_files_ignores_non_markdown_and_hashes_content(tmp_path: Path):
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("ignored", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.md").write_text("world", encoding="utf-8")

    snapshots = scan_markdown_files(tmp_path)
    assert [s.relative_path for s in snapshots] == ["a.md", "nested/c.md"]
    assert [s.source_file for s in snapshots] == ["a.md", "c.md"]
    assert all(len(s.content_hash) == 64 for s in snapshots)


def test_detect_changes_added_unchanged_modified_and_deleted(tmp_path: Path):
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    registry.upsert_document(source_file="same.md", relative_path="same.md", content_hash="same-hash")
    registry.upsert_document(source_file="changed.md", relative_path="changed.md", content_hash="old-hash")
    registry.upsert_document(source_file="deleted.md", relative_path="deleted.md", content_hash="deleted-hash")

    snapshots = [
        type("S", (), {"source_file": "same.md", "relative_path": "same.md", "content_hash": "same-hash", "size_bytes": 1})(),
        type("S", (), {"source_file": "changed.md", "relative_path": "changed.md", "content_hash": "new-hash", "size_bytes": 1})(),
        type("S", (), {"source_file": "added.md", "relative_path": "added.md", "content_hash": "added-hash", "size_bytes": 1})(),
    ]

    changes = detect_changes(registry, snapshots)
    assert [s.source_file for s in changes.added] == ["added.md"]
    assert [s.source_file for s in changes.modified] == ["changed.md"]
    assert [s.source_file for s in changes.unchanged] == ["same.md"]
    assert changes.deleted == ["deleted.md"]
    assert changes.to_summary() == {"added": 1, "modified": 1, "deleted": 1, "unchanged": 1, "total_changed": 3}


def test_empty_registry_marks_all_markdown_as_added(tmp_path: Path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")

    changes = detect_changes(registry, scan_markdown_files(tmp_path))
    assert [s.source_file for s in changes.added] == ["a.md", "b.md"]
    assert changes.modified == []
    assert changes.deleted == []
    assert changes.unchanged == []
