"""SQLite document registry for Stage2 markdown synchronization.

The registry is intentionally small: it only tracks document file hashes and
chunk-to-vector-point mappings.  Sync run logs and generic key-value state are
left out to keep Stage2 convergent for a single-project RAG knowledge base.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    source_file: str
    relative_path: str
    content_hash: str
    status: str
    chunk_count: int
    updated_at: str
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChunkMapping:
    chunk_id: str
    source_file: str
    chunk_index: int
    content_hash: str
    qdrant_point_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DocumentRegistry:
    """Minimal SQLite registry for detecting and applying document changes."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with closing(self.connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    source_file TEXT PRIMARY KEY,
                    relative_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT
                );

                CREATE TABLE IF NOT EXISTS chunk_mappings (
                    chunk_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    qdrant_point_id TEXT NOT NULL,
                    FOREIGN KEY(source_file) REFERENCES documents(source_file)
                );

                CREATE INDEX IF NOT EXISTS idx_chunk_mappings_source_file
                ON chunk_mappings(source_file);
                """
            )

    def upsert_document(
        self,
        *,
        source_file: str,
        relative_path: str,
        content_hash: str,
        status: str = "active",
        chunk_count: int = 0,
        last_error: str | None = None,
    ) -> DocumentRecord:
        self.init_schema()
        record = DocumentRecord(
            source_file=source_file,
            relative_path=relative_path,
            content_hash=content_hash,
            status=status,
            chunk_count=int(chunk_count),
            updated_at=utc_now_iso(),
            last_error=last_error,
        )
        with closing(self.connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO documents(source_file, relative_path, content_hash, status, chunk_count, updated_at, last_error)
                VALUES(:source_file, :relative_path, :content_hash, :status, :chunk_count, :updated_at, :last_error)
                ON CONFLICT(source_file) DO UPDATE SET
                    relative_path=excluded.relative_path,
                    content_hash=excluded.content_hash,
                    status=excluded.status,
                    chunk_count=excluded.chunk_count,
                    updated_at=excluded.updated_at,
                    last_error=excluded.last_error
                """,
                record.to_dict(),
            )
        return record

    def get_document(self, source_file: str) -> DocumentRecord | None:
        self.init_schema()
        with closing(self.connect()) as conn, conn:
            row = conn.execute("SELECT * FROM documents WHERE source_file = ?", (source_file,)).fetchone()
        return _document_from_row(row) if row else None

    def list_documents(self, *, include_deleted: bool = True) -> list[DocumentRecord]:
        self.init_schema()
        sql = "SELECT * FROM documents"
        params: tuple[Any, ...] = ()
        if not include_deleted:
            sql += " WHERE status != ?"
            params = ("deleted",)
        sql += " ORDER BY source_file"
        with closing(self.connect()) as conn, conn:
            rows = conn.execute(sql, params).fetchall()
        return [_document_from_row(row) for row in rows]

    def mark_deleted(self, source_file: str) -> None:
        self.init_schema()
        with closing(self.connect()) as conn, conn:
            existing = conn.execute("SELECT source_file FROM documents WHERE source_file = ?", (source_file,)).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE documents
                    SET status = ?, chunk_count = 0, updated_at = ?
                    WHERE source_file = ?
                    """,
                    ("deleted", utc_now_iso(), source_file),
                )
            conn.execute("DELETE FROM chunk_mappings WHERE source_file = ?", (source_file,))

    def replace_chunk_mappings(self, source_file: str, mappings: Iterable[ChunkMapping]) -> None:
        self.init_schema()
        rows = [mapping.to_dict() for mapping in mappings]
        for row in rows:
            if row["source_file"] != source_file:
                raise ValueError("all chunk mappings must have the same source_file")
        with closing(self.connect()) as conn, conn:
            conn.execute("DELETE FROM chunk_mappings WHERE source_file = ?", (source_file,))
            conn.executemany(
                """
                INSERT INTO chunk_mappings(chunk_id, source_file, chunk_index, content_hash, qdrant_point_id)
                VALUES(:chunk_id, :source_file, :chunk_index, :content_hash, :qdrant_point_id)
                """,
                rows,
            )
            conn.execute(
                "UPDATE documents SET chunk_count = ?, updated_at = ? WHERE source_file = ?",
                (len(rows), utc_now_iso(), source_file),
            )

    def list_chunk_mappings(self, source_file: str) -> list[ChunkMapping]:
        self.init_schema()
        with closing(self.connect()) as conn, conn:
            rows = conn.execute(
                "SELECT * FROM chunk_mappings WHERE source_file = ? ORDER BY chunk_index",
                (source_file,),
            ).fetchall()
        return [_mapping_from_row(row) for row in rows]

    def document_hashes(self, *, active_only: bool = True) -> dict[str, str]:
        self.init_schema()
        sql = "SELECT source_file, content_hash FROM documents"
        params: tuple[Any, ...] = ()
        if active_only:
            sql += " WHERE status != ?"
            params = ("deleted",)
        with closing(self.connect()) as conn, conn:
            rows = conn.execute(sql, params).fetchall()
        return {str(row["source_file"]): str(row["content_hash"]) for row in rows}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        source_file=str(row["source_file"]),
        relative_path=str(row["relative_path"]),
        content_hash=str(row["content_hash"]),
        status=str(row["status"]),
        chunk_count=int(row["chunk_count"]),
        updated_at=str(row["updated_at"]),
        last_error=row["last_error"],
    )


def _mapping_from_row(row: sqlite3.Row) -> ChunkMapping:
    return ChunkMapping(
        chunk_id=str(row["chunk_id"]),
        source_file=str(row["source_file"]),
        chunk_index=int(row["chunk_index"]),
        content_hash=str(row["content_hash"]),
        qdrant_point_id=str(row["qdrant_point_id"]),
    )
