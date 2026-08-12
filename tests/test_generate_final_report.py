from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_final_report import build_report


def result_row(query_id, *, metric=1.0, disaster="flood"):
    return {
        "query_id": query_id,
        "query": "洪涝灾害怎么办",
        "query_type": "scenario",
        "disaster_type": disaster,
        "status": "ok",
        "composer_mode": "deepseek",
        "is_fallback": disaster == "out_of_domain",
        "expected_fallback": True if disaster == "out_of_domain" else False,
        "all_citations_mapped": True,
        "unknown_citation_ids": [],
        "faithfulness": metric,
        "answer_relevancy": metric,
        "citation_correctness": metric,
        "citation_completeness": metric,
        "latency_ms": 100,
        "retrieval_metrics": {
            "doc_recall@5": metric,
            "doc_recall@10": metric,
            "chunk_recall@5": metric,
            "chunk_recall@10": metric,
            "doc_mrr@10": metric,
            "chunk_mrr@10": metric,
            "doc_ndcg@10": metric,
            "chunk_ndcg@10": metric,
            "latency_ms": 50,
        },
    }


def test_build_report_separates_labeled_and_random_claims():
    legacy = {"results": [result_row("legacy", metric=0.5)]}
    labeled = {"results": [result_row("labeled", metric=1.0)]}
    random = {"results": [result_row("random"), result_row("ood", disaster="out_of_domain")]}
    metrics, markdown = build_report(legacy, labeled, random)
    assert metrics["v2_labeled_retrieval"]["doc_recall@10"] == 1.0
    assert metrics["deterministic"]["random_ood_fallback_accuracy"] == 1.0
    assert "随机鲁棒性集没有人工相关性标签，不报告 Recall" in markdown
    assert "Legacy vs V2 Full" in markdown


def test_build_report_includes_ood_confusion_and_before_snapshot():
    legacy = {"results": [result_row("legacy", metric=0.5)]}
    labeled = {"results": [result_row("labeled")]}
    random = {"results": [result_row("ood", disaster="out_of_domain")]}
    metrics, markdown = build_report(legacy, labeled, random, {"stage": "before"})
    matrix = metrics["decision_metrics"]["confusion_matrix"]
    assert matrix["ood_fallback"] == 1
    assert metrics["before_fix"] == {"stage": "before"}
    assert "Before/After" in markdown
