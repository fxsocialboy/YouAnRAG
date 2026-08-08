"""Metadata and artifact helpers for stage 1 chunks.

Stage 1.5 turns in-memory Chunk objects into stable JSON artifacts used by the
next indexing stage.  It keeps the implementation intentionally small: no UUID5,
no document-version registry, and no database writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from rag_v2.config import Stage1ChunkParams
from rag_v2.schemas import Chunk


@dataclass(slots=True)
class ChunkQualityReport:
    total_chunks: int
    indexable_chunks: int
    unique_source_files: int
    max_tokens: int
    avg_tokens: float
    max_chars: int
    avg_chars: float
    over_hard_max_chunks: int
    missing_section_path_chunks: int
    missing_section_path_ratio: float
    duplicate_chunks: int
    duplicate_ratio: float
    hard_max_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def enrich_chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    """Return JSON-serializable metadata for one chunk.

    The output is deliberately close to the legacy ``chunk_metadata.json`` shape
    while adding fields needed for debugging and source display.
    """

    data = chunk.to_dict()
    data["content_hash"] = sha256_text(chunk.content)
    data["embedding_text_hash"] = sha256_text(chunk.embedding_text)
    data["section_path_text"] = " > ".join(chunk.section_path)
    return data


def build_quality_report(chunks: list[Chunk], params: Stage1ChunkParams) -> ChunkQualityReport:
    total = len(chunks)
    indexable = [chunk for chunk in chunks if chunk.is_indexable]
    token_counts = [chunk.token_count for chunk in indexable]
    char_counts = [chunk.char_count for chunk in indexable]
    duplicate_chunks = count_duplicate_contents(indexable)
    missing_section = sum(1 for chunk in indexable if not chunk.section_path)
    n = len(indexable) or 1
    return ChunkQualityReport(
        total_chunks=total,
        indexable_chunks=len(indexable),
        unique_source_files=len({chunk.source_file for chunk in chunks}),
        max_tokens=max(token_counts, default=0),
        avg_tokens=round(sum(token_counts) / n, 2),
        max_chars=max(char_counts, default=0),
        avg_chars=round(sum(char_counts) / n, 2),
        over_hard_max_chunks=sum(1 for chunk in indexable if chunk.token_count > params.hard_max_tokens),
        missing_section_path_chunks=missing_section,
        missing_section_path_ratio=round(missing_section / n, 4),
        duplicate_chunks=duplicate_chunks,
        duplicate_ratio=round(duplicate_chunks / n, 4),
        hard_max_tokens=params.hard_max_tokens,
    )


def count_duplicate_contents(chunks: Iterable[Chunk]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for chunk in chunks:
        key = sha256_text(chunk.content.strip())
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
