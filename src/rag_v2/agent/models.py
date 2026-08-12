"""Agent-facing contracts for Stage6 answer generation.

Stage6 keeps the retrieval pipeline isolated from downstream Agent code by
converting packed evidence, citations and trace data into lightweight,
JSON-ready dataclasses.  These models intentionally avoid importing heavy
retrieval/model dependencies so they can be used by CLI, FastAPI and unit tests
without loading BGE, Qdrant or reranker models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clean_section_path(section_path: list[str] | tuple[str, ...] | None) -> list[str]:
    if section_path is None:
        return []
    if not isinstance(section_path, (list, tuple)):
        raise TypeError("section_path must be a list[str]")
    return [str(item).strip() for item in section_path if str(item).strip()]


def _normalize_citation_id(citation_id: str) -> str:
    value = str(citation_id).strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    if not value:
        raise ValueError("citation_id must not be empty")
    return value


@dataclass(slots=True)
class EvidenceItem:
    """A retrieved chunk promoted to answer evidence and assigned a citation id."""

    citation_id: str
    chunk_id: str
    source_file: str
    section_path: list[str]
    content: str
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.citation_id = _normalize_citation_id(self.citation_id)
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if not self.source_file:
            raise ValueError("source_file must not be empty")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        self.section_path = _clean_section_path(self.section_path)
        self.content = str(self.content)
        self.score = float(self.score)
        self.metadata = dict(self.metadata or {})

    @property
    def marker(self) -> str:
        return f"[{self.citation_id}]"

    @property
    def section_path_text(self) -> str:
        return " / ".join(self.section_path)

    def to_citation(self) -> "Citation":
        return Citation(
            citation_id=self.citation_id,
            source_file=self.source_file,
            chunk_id=self.chunk_id,
            section_path=self.section_path,
            rank=self.rank,
            score=self.score,
            metadata={
                "content_preview": self.content[:160],
                **self.metadata,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["marker"] = self.marker
        data["section_path_text"] = self.section_path_text
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceItem":
        payload = dict(data)
        payload.pop("marker", None)
        payload.pop("section_path_text", None)
        return cls(**payload)


@dataclass(slots=True)
class Citation:
    """A compact source reference exposed to answer users and Agent nodes."""

    citation_id: str
    source_file: str
    chunk_id: str
    section_path: list[str] = field(default_factory=list)
    rank: int = 0
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.citation_id = _normalize_citation_id(self.citation_id)
        if not self.source_file:
            raise ValueError("source_file must not be empty")
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        self.section_path = _clean_section_path(self.section_path)
        self.score = float(self.score)
        self.metadata = dict(self.metadata or {})

    @property
    def marker(self) -> str:
        return f"[{self.citation_id}]"

    @property
    def section_path_text(self) -> str:
        return " / ".join(self.section_path)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["marker"] = self.marker
        data["section_path_text"] = self.section_path_text
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Citation":
        payload = dict(data)
        payload.pop("marker", None)
        payload.pop("section_path_text", None)
        return cls(**payload)


@dataclass(slots=True)
class AnswerTrace:
    """Debug trace returned with Stage6 answers for inspection and demos."""

    query_plan: dict[str, Any] = field(default_factory=dict)
    branches: list[dict[str, Any]] = field(default_factory=list)
    matched_branches: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    composer_mode: str = "template"
    verification: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query_plan = dict(self.query_plan or {})
        self.branches = [dict(item) for item in (self.branches or [])]
        self.matched_branches = [str(item) for item in (self.matched_branches or [])]
        self.retrieval_latency_ms = float(self.retrieval_latency_ms)
        self.composer_mode = str(self.composer_mode or "template")
        self.verification = dict(self.verification or {})
        self.extra = dict(self.extra or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnswerTrace":
        return cls(**dict(data))


@dataclass(slots=True)
class RagAnswer:
    """Final Stage6 response object returned by Agent-facing APIs."""

    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    trace: AnswerTrace = field(default_factory=AnswerTrace)
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        self.query = str(self.query)
        self.answer = str(self.answer)
        self.citations = [item if isinstance(item, Citation) else Citation.from_dict(item) for item in self.citations]
        self.evidence = [item if isinstance(item, EvidenceItem) else EvidenceItem.from_dict(item) for item in self.evidence]
        if not isinstance(self.trace, AnswerTrace):
            self.trace = AnswerTrace.from_dict(self.trace)
        if self.fallback_reason is not None:
            self.fallback_reason = str(self.fallback_reason)

    @property
    def is_fallback(self) -> bool:
        return self.fallback_reason is not None

    @property
    def decision(self) -> str:
        return "fallback" if self.is_fallback else "answered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [item.to_dict() for item in self.citations],
            "evidence": [item.to_dict() for item in self.evidence],
            "trace": self.trace.to_dict(),
            "fallback_reason": self.fallback_reason,
            "is_fallback": self.is_fallback,
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagAnswer":
        payload = dict(data)
        payload.pop("is_fallback", None)
        payload.pop("decision", None)
        return cls(**payload)
