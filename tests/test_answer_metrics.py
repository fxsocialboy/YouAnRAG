import pytest

from rag_v2.evaluation.answer_metrics import (
    citation_mapping_metrics,
    fallback_accuracy,
    latency_summary,
    summarize_answer_rows,
)
from rag_v2.evaluation.models import AnswerJudgeResult, AtomicFactJudgment


def test_citation_mapping_detects_unknown_markers():
    metrics = citation_mapping_metrics("措施一[S1]，措施二[S999]。", ["S1", "S2"])

    assert metrics["mapped_citation_ids"] == ["S1"]
    assert metrics["unknown_citation_ids"] == ["S999"]
    assert metrics["citation_mapping_ratio"] == 0.5
    assert metrics["all_citations_mapped"] is False


def test_fallback_accuracy_only_uses_labeled_rows():
    rows = [
        {"expected_fallback": True, "is_fallback": True},
        {"expected_fallback": False, "is_fallback": True},
        {"expected_fallback": None, "is_fallback": False},
    ]
    assert fallback_accuracy(rows) == 0.5


def test_answer_judge_result_aggregates_atomic_facts():
    result = AnswerJudgeResult(
        composer_mode="deepseek",
        atomic_facts=[
            AtomicFactJudgment("事实一", supported=True, cited=True, supporting_citation_ids=["S1"]),
            AtomicFactJudgment("事实二", supported=False, cited=True, supporting_citation_ids=["S2"]),
            AtomicFactJudgment("事实三", supported=True, cited=False),
        ],
        answer_relevancy=0.8,
    )

    assert result.faithfulness == pytest.approx(2 / 3)
    assert result.citation_completeness == pytest.approx(2 / 3)
    assert result.citation_correctness == 0.5
    assert result.to_dict()["answer_relevancy"] == 0.8


def test_summary_keeps_composer_modes_separate():
    rows = [
        {
            "composer_mode": "deepseek",
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "all_citations_mapped": True,
            "is_fallback": False,
            "expected_fallback": False,
            "latency_ms": 100,
        },
        {
            "composer_mode": "template",
            "faithfulness": 1.0,
            "answer_relevancy": None,
            "all_citations_mapped": True,
            "is_fallback": False,
            "expected_fallback": False,
            "latency_ms": 20,
        },
    ]
    deepseek = summarize_answer_rows(rows, composer_mode="deepseek")

    assert deepseek["query_count"] == 1
    assert deepseek["faithfulness"] == 0.8
    assert deepseek["avg_latency_ms"] == 100.0


def test_latency_summary_interpolates_percentiles():
    summary = latency_summary([10, 20, 30, 40])
    assert summary["p50_latency_ms"] == 25.0
    assert summary["p95_latency_ms"] == 38.5


def test_summary_excludes_fallback_rows_from_answer_quality_and_reports_coverage():
    rows = [
        {
            "composer_mode": "deepseek", "faithfulness": 1.0, "answer_relevancy": 0.8,
            "all_citations_mapped": True, "is_fallback": False, "fallback_reason": None,
            "expected_fallback": False, "judge": {"faithfulness": 1.0}, "latency_ms": 10,
        },
        {
            "composer_mode": "deepseek", "faithfulness": 0.0, "answer_relevancy": 0.0,
            "all_citations_mapped": True, "is_fallback": True, "fallback_reason": "out_of_domain",
            "expected_fallback": True, "judge": None, "latency_ms": 10,
        },
    ]
    summary = summarize_answer_rows(rows)
    assert summary["faithfulness"] == 1.0
    assert summary["answer_relevancy"] == 0.8
    assert summary["fallback_reason_consistency"] == 1.0
    assert summary["judge_coverage"] == 1.0
