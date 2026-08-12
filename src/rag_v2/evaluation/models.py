"""Stage7 evaluation data contracts.

These models do not import retrieval/model dependencies.  Dataset validation
can therefore run in unit tests and before an expensive AutoDL evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


ALLOWED_QUERY_TYPES = {
    "scenario",
    "exact_fact",
    "keyword",
    "short_ambiguous",
    "multi_hop",
    "out_of_domain",
}
ALLOWED_DISASTER_TYPES = {
    "meteorological",
    "flood",
    "geological",
    "earthquake",
    "comprehensive",
    "out_of_domain",
}
ALLOWED_REGRESSION_DECISIONS = {"answered", "fallback"}
ALLOWED_REGRESSION_REASON_CATEGORIES = {
    "in_domain",
    "out_of_domain",
    "insufficient_evidence",
}
ALLOWED_REGRESSION_CATEGORIES = {
    "ood_leak",
    "false_rejection",
    "positive_control",
    "domain_boundary",
}


def _clean_strings(values: list[str] | tuple[str, ...] | None) -> list[str]:
    return [str(value).strip() for value in (values or []) if str(value).strip()]


@dataclass(slots=True)
class EvaluationQuery:
    query_id: str
    query: str
    query_type: str
    disaster_type: str
    relevant_source_files: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    reference_facts: list[str] = field(default_factory=list)
    expected_fallback: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query_id = str(self.query_id).strip()
        self.query = str(self.query).strip()
        self.query_type = str(self.query_type).strip()
        self.disaster_type = str(self.disaster_type).strip()
        self.relevant_source_files = _clean_strings(self.relevant_source_files)
        self.relevant_chunk_ids = _clean_strings(self.relevant_chunk_ids)
        self.reference_facts = _clean_strings(self.reference_facts)
        self.metadata = dict(self.metadata or {})
        self.validate()

    def validate(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must not be empty")
        if not self.query:
            raise ValueError(f"{self.query_id}: query must not be empty")
        if self.query_type not in ALLOWED_QUERY_TYPES:
            raise ValueError(f"{self.query_id}: unsupported query_type={self.query_type!r}")
        if self.disaster_type not in ALLOWED_DISASTER_TYPES:
            raise ValueError(f"{self.query_id}: unsupported disaster_type={self.disaster_type!r}")
        if self.expected_fallback is False:
            if not self.relevant_source_files or not self.relevant_chunk_ids:
                raise ValueError(f"{self.query_id}: answerable query requires relevant sources and chunks")
            if not self.reference_facts:
                raise ValueError(f"{self.query_id}: answerable query requires reference_facts")
        if self.expected_fallback is True and (self.relevant_source_files or self.relevant_chunk_ids):
            raise ValueError(f"{self.query_id}: fallback query must not carry relevant labels")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationQuery":
        return cls(**dict(data))


@dataclass(slots=True)
class Stage74RegressionQuery:
    """A Stage 7.4 regression row plus its frozen-source provenance."""

    query_id: str
    query: str
    query_type: str
    disaster_type: str
    expected_decision: str
    expected_reason_category: str
    regression_category: str
    source_dataset: str
    source_query_id: str
    relevant_source_files: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    reference_facts: list[str] = field(default_factory=list)
    expected_fallback: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expected_decision = str(self.expected_decision).strip()
        self.expected_reason_category = str(self.expected_reason_category).strip()
        self.regression_category = str(self.regression_category).strip()
        self.source_dataset = str(self.source_dataset).strip()
        self.source_query_id = str(self.source_query_id).strip()
        # Reuse the frozen evaluation contract instead of maintaining a second
        # implementation for query fields and relevance labels.
        base = self.to_evaluation_query()
        if self.expected_decision not in ALLOWED_REGRESSION_DECISIONS:
            raise ValueError(f"{base.query_id}: invalid expected_decision={self.expected_decision!r}")
        if self.expected_reason_category not in ALLOWED_REGRESSION_REASON_CATEGORIES:
            raise ValueError(
                f"{base.query_id}: invalid expected_reason_category={self.expected_reason_category!r}"
            )
        if self.regression_category not in ALLOWED_REGRESSION_CATEGORIES:
            raise ValueError(f"{base.query_id}: invalid regression_category={self.regression_category!r}")
        if not self.source_dataset or not self.source_query_id:
            raise ValueError(f"{base.query_id}: source provenance must not be empty")
        should_fallback = self.expected_decision == "fallback"
        if self.expected_fallback is not should_fallback:
            raise ValueError(
                f"{base.query_id}: expected_fallback must agree with expected_decision"
            )

    def to_evaluation_query(self) -> EvaluationQuery:
        # Unlabeled random/boundary rows still have a regression-level
        # decision, but they intentionally do not pretend to have retrieval
        # ground truth in the generic evaluator contract.
        evaluator_expected_fallback = self.expected_fallback
        if self.expected_fallback is False and not self.relevant_chunk_ids:
            evaluator_expected_fallback = None
        return EvaluationQuery(
            query_id=self.query_id,
            query=self.query,
            query_type=self.query_type,
            disaster_type=self.disaster_type,
            relevant_source_files=self.relevant_source_files,
            relevant_chunk_ids=self.relevant_chunk_ids,
            reference_facts=self.reference_facts,
            expected_fallback=evaluator_expected_fallback,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Stage74RegressionQuery":
        return cls(**dict(data))


@dataclass(slots=True)
class AtomicFactJudgment:
    fact: str
    supported: bool
    cited: bool
    supporting_citation_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def __post_init__(self) -> None:
        self.fact = str(self.fact).strip()
        if not self.fact:
            raise ValueError("fact must not be empty")
        self.supporting_citation_ids = _clean_strings(self.supporting_citation_ids)
        self.reason = str(self.reason).strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerJudgeResult:
    composer_mode: str
    atomic_facts: list[AtomicFactJudgment] = field(default_factory=list)
    answer_relevancy: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        self.composer_mode = str(self.composer_mode).strip()
        if self.composer_mode not in {"template", "deepseek"}:
            raise ValueError("composer_mode must be template or deepseek")
        self.atomic_facts = [
            item if isinstance(item, AtomicFactJudgment) else AtomicFactJudgment(**item)
            for item in self.atomic_facts
        ]
        if self.answer_relevancy is not None and not 0.0 <= float(self.answer_relevancy) <= 1.0:
            raise ValueError("answer_relevancy must be within [0, 1]")
        if self.answer_relevancy is not None:
            self.answer_relevancy = float(self.answer_relevancy)
        self.reason = str(self.reason).strip()

    @property
    def faithfulness(self) -> float | None:
        if not self.atomic_facts:
            return None
        return sum(item.supported for item in self.atomic_facts) / len(self.atomic_facts)

    @property
    def citation_completeness(self) -> float | None:
        if not self.atomic_facts:
            return None
        return sum(item.cited for item in self.atomic_facts) / len(self.atomic_facts)

    @property
    def citation_correctness(self) -> float | None:
        cited_facts = [item for item in self.atomic_facts if item.cited]
        if not cited_facts:
            return None
        return sum(item.supported for item in cited_facts) / len(cited_facts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "composer_mode": self.composer_mode,
            "atomic_facts": [item.to_dict() for item in self.atomic_facts],
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "citation_completeness": self.citation_completeness,
            "citation_correctness": self.citation_correctness,
            "reason": self.reason,
        }


def load_evaluation_queries(path: str | Path) -> list[EvaluationQuery]:
    source = Path(path)
    rows: list[EvaluationQuery] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(EvaluationQuery.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
    return rows


def load_stage74_regression_queries(path: str | Path) -> list[Stage74RegressionQuery]:
    source = Path(path)
    rows: list[Stage74RegressionQuery] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(Stage74RegressionQuery.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
    return rows


def validate_evaluation_dataset(
    queries: list[EvaluationQuery],
    *,
    known_chunk_ids: set[str] | None = None,
    known_source_files: set[str] | None = None,
) -> dict[str, Any]:
    ids = [item.query_id for item in queries]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    duplicate_queries = sorted({item.query for item in queries if sum(q.query == item.query for q in queries) > 1})
    unknown_chunks: list[str] = []
    unknown_sources: list[str] = []
    if known_chunk_ids is not None:
        unknown_chunks = sorted({chunk for item in queries for chunk in item.relevant_chunk_ids if chunk not in known_chunk_ids})
    if known_source_files is not None:
        unknown_sources = sorted(
            {source for item in queries for source in item.relevant_source_files if source not in known_source_files}
        )
    group_counts: dict[str, int] = {}
    disaster_type_counts: dict[str, int] = {}
    for item in queries:
        group_counts[item.query_type] = group_counts.get(item.query_type, 0) + 1
        disaster_type_counts[item.disaster_type] = disaster_type_counts.get(item.disaster_type, 0) + 1
    errors = {
        "duplicate_query_ids": duplicate_ids,
        "duplicate_queries": duplicate_queries,
        "unknown_chunk_ids": unknown_chunks,
        "unknown_source_files": unknown_sources,
    }
    return {
        "query_count": len(queries),
        "group_counts": group_counts,
        "disaster_type_counts": disaster_type_counts,
        **errors,
        "valid": not any(errors.values()),
    }
