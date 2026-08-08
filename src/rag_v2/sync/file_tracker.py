"""Markdown file scanning and change detection for Stage2 sync."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from rag_v2.sync.registry import DocumentRegistry


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    source_file: str
    relative_path: str
    content_hash: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class FileChangeSet:
    added: list[FileSnapshot] = field(default_factory=list)
    modified: list[FileSnapshot] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[FileSnapshot] = field(default_factory=list)

    def total_changed(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)

    def to_summary(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "modified": len(self.modified),
            "deleted": len(self.deleted),
            "unchanged": len(self.unchanged),
            "total_changed": self.total_changed(),
        }


def scan_markdown_files(markdown_dir: str | Path) -> list[FileSnapshot]:
    root = Path(markdown_dir)
    if not root.exists():
        raise FileNotFoundError(f"markdown dir does not exist: {root}")
    snapshots: list[FileSnapshot] = []
    for path in sorted(root.rglob("*.md")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        snapshots.append(
            FileSnapshot(
                source_file=path.name,
                relative_path=relative_path,
                content_hash=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return snapshots


def detect_changes(registry: DocumentRegistry, snapshots: list[FileSnapshot]) -> FileChangeSet:
    known_hashes = registry.document_hashes(active_only=True)
    current_by_source = {snapshot.source_file: snapshot for snapshot in snapshots}

    added: list[FileSnapshot] = []
    modified: list[FileSnapshot] = []
    unchanged: list[FileSnapshot] = []

    for snapshot in snapshots:
        old_hash = known_hashes.get(snapshot.source_file)
        if old_hash is None:
            added.append(snapshot)
        elif old_hash != snapshot.content_hash:
            modified.append(snapshot)
        else:
            unchanged.append(snapshot)

    deleted = sorted(source_file for source_file in known_hashes if source_file not in current_by_source)
    return FileChangeSet(added=added, modified=modified, deleted=deleted, unchanged=unchanged)


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()
