"""Evidence sufficiency guardrail for Stage6 answers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_v2.agent.models import EvidenceItem
from rag_v2.agent.verifier import VerificationResult


@dataclass(slots=True)
class EvidenceGuardrailConfig:
    min_evidence_count: int = 1
    min_score: float = 0.3
    fallback_answer: str = "当前知识库没有足够依据回答该问题。"

    def validate(self) -> None:
        if self.min_evidence_count < 0:
            raise ValueError("min_evidence_count must be non-negative")
        if self.min_score < 0:
            raise ValueError("min_score must be non-negative")


@dataclass(slots=True)
class GuardrailResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    fallback_answer: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "fallback_answer": self.fallback_answer,
            "metrics": self.metrics,
        }


class EvidenceGuardrail:
    """Decide whether the answer has enough reliable evidence to be returned."""

    def __init__(self, config: EvidenceGuardrailConfig | None = None):
        self.config = config or EvidenceGuardrailConfig()
        self.config.validate()

    def check(self, evidence: list[EvidenceItem], verification: VerificationResult | None = None) -> GuardrailResult:
        reasons: list[str] = []
        evidence_count = len(evidence)
        confidence_items = [
            (item, _item_retrieval_confidence(item))
            for item in evidence
            if _item_retrieval_confidence(item) is not None
        ]
        max_score = max((value for _item, value in confidence_items), default=None)
        confidence_type = "unavailable"
        if confidence_items:
            best, _value = max(confidence_items, key=lambda pair: pair[1])
            confidence_type = str(best.metadata.get("confidence_type", "legacy"))

        if evidence_count < self.config.min_evidence_count:
            reasons.append("insufficient_evidence")
        if max_score is not None and max_score < self.config.min_score:
            reasons.append("low_retrieval_score")
        if verification is not None and not verification.passed:
            reasons.extend(reason for reason in verification.reasons if reason not in reasons)

        passed = not reasons
        return GuardrailResult(
            passed=passed,
            reasons=reasons,
            fallback_answer=None if passed else self.config.fallback_answer,
            metrics={
                "evidence_count": evidence_count,
                "max_score": max_score,
                "confidence_value": max_score,
                "confidence_type": confidence_type,
                "confidence_state": _confidence_state(max_score),
                "min_score": self.config.min_score,
                "min_evidence_count": self.config.min_evidence_count,
            },
        )


def _confidence_state(value: float | None) -> str:
    if value is None:
        return "confidence_unavailable"
    if value < 0.20:
        return "low_confidence"
    if value < 0.50:
        return "uncertain"
    return "retrieval_confident"


def _item_retrieval_confidence(item: EvidenceItem) -> float | None:
    if item.metadata.get("confidence_type") == "unavailable":
        return None
    value = item.metadata.get("retrieval_confidence", item.score)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
