"""Data contracts for Stage5 query planning.

These dataclasses are deliberately lightweight and dependency-free.  Stage5
uses them as a stable boundary between query analysis, rewrite, multi-query
retrieval and optional HyDE branches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

QueryType = Literal["keyword", "exact_fact", "scenario", "multi_hop", "ambiguous"]


@dataclass(slots=True, frozen=True)
class RetrievalPolicy:
    """Routing knobs chosen by QueryAnalyzer.

    Defaults are conservative: raw query is always preserved, HyDE is disabled,
    and Stage4 rerank/MMR stay configurable instead of being forced globally.
    """

    use_rewrite: bool = False
    use_multi_query: bool = False
    use_hyde: bool = False
    use_reranker: bool = True
    use_mmr: bool = True
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rerank_top_k: int = 30
    max_branches: int = 3
    branch_weights: dict[str, float] = field(
        default_factory=lambda: {"raw": 1.0, "normalized": 0.9, "expanded": 0.7, "hyde": 0.6}
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class QueryBranch:
    """A retrieval branch derived from the original query."""

    branch: str
    query: str
    weight: float
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class QueryPlan:
    """Structured output of Stage5 QueryAnalyzer."""

    original_query: str
    normalized_query: str
    query_type: QueryType
    region_hint: str | None = None
    hazard_hint: str | None = None
    document_type_hint: str | None = None
    time_hint: str | None = None
    extracted_terms: list[str] = field(default_factory=list)
    rewrite_confidence: float = 0.5
    retrieval_policy: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        return data
