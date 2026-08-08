"""Lightweight shared schemas for stage 1 RAG V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    """A single indexable or non-indexable chunk produced by the stage 1 pipeline."""

    chunk_id: str
    source_file: str
    chunk_index: int
    section_path: list[str]
    content: str
    embedding_text: str
    token_count: int
    char_count: int
    is_indexable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if not self.source_file:
            raise ValueError("source_file must not be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.token_count < 0 or self.char_count < 0:
            raise ValueError("token_count and char_count must be non-negative")
        if not isinstance(self.section_path, list):
            raise TypeError("section_path must be a list[str]")
        self.section_path = [str(item) for item in self.section_path if str(item).strip()]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass(slots=True)
class RetrievedChunk:
    """Search result used by V2 while preserving legacy adapter compatibility."""

    chunk: Chunk
    score: float
    rank: int

    def to_legacy_content(self) -> str:
        return self.chunk.content

    def to_dict(self) -> dict[str, Any]:
        return {"chunk": self.chunk.to_dict(), "score": self.score, "rank": self.rank}
