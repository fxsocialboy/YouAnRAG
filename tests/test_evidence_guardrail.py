import pytest

from rag_v2.agent.guardrail import EvidenceGuardrail, EvidenceGuardrailConfig
from rag_v2.agent.models import EvidenceItem
from rag_v2.agent.verifier import VerificationResult


def make_evidence(score=0.8):
    return EvidenceItem(
        citation_id="S1",
        chunk_id="a.md::1",
        source_file="a.md",
        section_path=[],
        content="content",
        score=score,
        rank=1,
    )


def test_guardrail_passes_with_enough_evidence_and_valid_verification():
    result = EvidenceGuardrail().check([make_evidence()], VerificationResult(passed=True))

    assert result.passed is True
    assert result.fallback_answer is None
    assert result.metrics["evidence_count"] == 1


def test_guardrail_fails_when_no_evidence():
    result = EvidenceGuardrail().check([], VerificationResult(passed=True))

    assert result.passed is False
    assert "insufficient_evidence" in result.reasons
    assert result.fallback_answer == "当前知识库没有足够依据回答该问题。"


def test_guardrail_fails_when_score_is_low():
    result = EvidenceGuardrail(EvidenceGuardrailConfig(min_score=0.3)).check([make_evidence(score=0.1)])

    assert result.passed is False
    assert "low_retrieval_score" in result.reasons
    assert result.metrics["max_score"] == pytest.approx(0.1)


def test_guardrail_merges_verifier_reasons():
    verification = VerificationResult(passed=False, reasons=["missing_citation"])
    result = EvidenceGuardrail().check([make_evidence()], verification)

    assert result.passed is False
    assert "missing_citation" in result.reasons


def test_guardrail_config_validates_thresholds():
    with pytest.raises(ValueError):
        EvidenceGuardrail(EvidenceGuardrailConfig(min_evidence_count=-1))
    with pytest.raises(ValueError):
        EvidenceGuardrail(EvidenceGuardrailConfig(min_score=-0.1))


def test_guardrail_uses_explicit_retrieval_confidence_not_mmr_score():
    evidence = make_evidence(score=0.1)
    evidence.metadata.update({"retrieval_confidence": 0.1, "confidence_type": "rerank", "mmr_score": 1.0})

    result = EvidenceGuardrail(EvidenceGuardrailConfig(min_score=0.3)).check([evidence])

    assert result.passed is False
    assert result.metrics["confidence_value"] == pytest.approx(0.1)
    assert result.metrics["confidence_type"] == "rerank"
    assert result.metrics["confidence_state"] == "low_confidence"


def test_guardrail_does_not_treat_rrf_as_probability():
    evidence = make_evidence(score=0.0)
    evidence.metadata.update({"confidence_type": "unavailable", "fusion_score": 0.03})

    result = EvidenceGuardrail(EvidenceGuardrailConfig(min_score=0.3)).check([evidence])

    assert result.passed is True
    assert result.metrics["confidence_value"] is None
    assert result.metrics["confidence_state"] == "confidence_unavailable"
