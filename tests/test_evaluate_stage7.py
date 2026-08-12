from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_stage7 import _load_existing, evaluate_legacy_row, evaluate_v2_row, summarize
from rag_v2.agent.models import AnswerTrace, EvidenceItem, RagAnswer
from rag_v2.agent.service import RagAnswerServiceOptions
from rag_v2.evaluation.models import AnswerJudgeResult, AtomicFactJudgment, EvaluationQuery


QUERY = EvaluationQuery(
    query_id="q1",
    query="台风黄色预警时学校怎么办？",
    query_type="scenario",
    disaster_type="meteorological",
    relevant_source_files=["policy.md"],
    relevant_chunk_ids=["policy.md::3"],
    reference_facts=["学校停止户外活动"],
    expected_fallback=False,
)


class FakeLegacy:
    def search(self, query, top_k=10):
        return [{"rank": 1, "source_file": "policy.md", "chunk_id": "policy.md::3", "score": 0.9}]


class FakeService:
    def answer(self, query, *, options=None):
        evidence = EvidenceItem("S1", "policy.md::3", "policy.md", ["学校"], "学校应停止户外活动。", 0.9, 1)
        return RagAnswer(
            query=query,
            answer="学校应停止户外活动。[S1]",
            citations=[evidence.to_citation()],
            evidence=[evidence],
            trace=AnswerTrace(
                retrieval_latency_ms=12.0,
                composer_mode="deepseek",
                extra={"composer": {"actual_mode": "deepseek"}},
            ),
        )


class FakeJudge:
    def evaluate(self, **kwargs):
        return AnswerJudgeResult(
            composer_mode="deepseek",
            atomic_facts=[AtomicFactJudgment("学校停止户外活动", True, True, ["S1"], "支持")],
            answer_relevancy=0.95,
        )


def test_legacy_row_calculates_labeled_retrieval_metrics():
    row = evaluate_legacy_row(QUERY, FakeLegacy(), top_k=10)
    assert row["status"] == "ok"
    assert row["retrieval_metrics"]["doc_recall@10"] == 1.0
    assert row["retrieval_metrics"]["chunk_mrr@10"] == 1.0


def test_v2_row_saves_answer_judge_citations_and_trace():
    row = evaluate_v2_row(
        QUERY,
        FakeService(),
        FakeJudge(),
        options=RagAnswerServiceOptions(composer_mode="deepseek"),
    )
    assert row["status"] == "ok"
    assert row["faithfulness"] == 1.0
    assert row["answer_relevancy"] == 0.95
    assert row["all_citations_mapped"] is True
    assert row["retrieval_metrics"]["chunk_recall@10"] == 1.0
    assert row["packed_evidence_results"][0]["chunk_id"] == "policy.md::3"
    assert row["online_answer_latency_ms"] == row["answer_latency_ms"]
    assert row["retrieval_metrics"]["latency_ms"] == 12.0
    assert row["evaluation_total_latency_ms"] == row["latency_ms"]


def test_summarize_counts_partial_and_errors():
    payload = {
        "query_count": 3,
        "backend": "v2",
        "results": [
            {"status": "ok", "composer_mode": "deepseek", "latency_ms": 1, "all_citations_mapped": True},
            {"status": "partial", "latency_ms": 2, "all_citations_mapped": True},
            {"status": "error", "latency_ms": 3, "all_citations_mapped": False},
        ],
    }
    result = summarize(payload)
    assert result["ok_count"] == 1
    assert result["partial_count"] == 1
    assert result["error_count"] == 1
    assert result["run_success_rate"] == 0.3333
    assert result["answer_rate"] == 1.0


def test_resume_requires_matching_config_hash(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"dataset":"labeled","backend":"v2","config_hash":"abc","results":[]}', encoding="utf-8")
    assert _load_existing(output, dataset="labeled", backend="v2", fresh=False, config_hash="abc")
    with pytest.raises(ValueError, match="config hash"):
        _load_existing(output, dataset="labeled", backend="v2", fresh=False, config_hash="different")


def test_random_unlabeled_retrieval_metrics_remain_na():
    query = EvaluationQuery(
        query_id="random", query="灾情如何上报", query_type="scenario", disaster_type="comprehensive"
    )
    row = evaluate_v2_row(query, FakeService(), None, options=RagAnswerServiceOptions(composer_mode="deepseek"))
    assert row["retrieval_metrics"]["doc_recall@10"] is None
    summary = summarize({"query_count": 1, "backend": "v2", "results": [row]})
    assert summary["retrieval"]["doc_recall@10"] is None
