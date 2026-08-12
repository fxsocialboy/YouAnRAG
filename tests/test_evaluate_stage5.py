from pathlib import Path

from evaluate_stage5 import summarize_rows
from scripts.generate_phase5_report import generate_phase5_report


def make_metrics(doc_hit=True, chunk_hit=True, doc_mrr=1.0, chunk_mrr=1.0):
    return {
        "latency_ms": 10.0,
        "doc_rank@10": 1 if doc_hit else None,
        "chunk_rank_exact@10": 1 if chunk_hit else None,
        "doc_hit@10": doc_hit,
        "chunk_hit_exact@10": chunk_hit,
        "doc_mrr@10": doc_mrr,
        "chunk_mrr_exact@10": chunk_mrr,
        "packed_nonempty": True,
        "packed_token_ratio": 0.5,
        "branch_count": 2,
        "hyde_used": False,
        "fallback": False,
        "doc_rank_shift_vs_stage4": 0,
        "chunk_rank_shift_vs_stage4": 0,
    }


def test_summarize_rows_computes_guardrail_no_regression():
    row = {
        "variant_metrics": {
            "stage4_raw": make_metrics(doc_hit=True, chunk_hit=True),
            "stage5_rewrite": make_metrics(doc_hit=True, chunk_hit=True),
            "stage5_multi_query": make_metrics(doc_hit=True, chunk_hit=False, chunk_mrr=0.0),
            "stage5_hyde": make_metrics(doc_hit=True, chunk_hit=True),
        }
    }

    summary = summarize_rows([row], top_k=10, guardrail=True)

    assert summary["variants"]["stage5_rewrite"]["guardrail_no_regression"] is True
    assert summary["variants"]["stage5_multi_query"]["guardrail_no_regression"] is False
    assert summary["variants"]["stage5_hyde"]["doc_recall@10"] == 1.0


def test_generate_phase5_report_from_minimal_eval(tmp_path, monkeypatch):
    project = tmp_path
    (project / "experiments").mkdir()
    variants = {
        name: {
            "doc_recall@10": 1.0,
            "chunk_recall_exact@10": 1.0,
            "doc_mrr@10": 1.0,
            "chunk_mrr_exact@10": 1.0,
            "doc_mrr_delta_vs_stage4": 0.0,
            "avg_branch_count": 1.0,
            "hyde_used_ratio": 0.0,
            "avg_latency_ms": 1.0,
            "guardrail_no_regression": True,
        }
        for name in ["stage4_raw", "stage5_rewrite", "stage5_multi_query", "stage5_hyde"]
    }
    payload = {
        "summary": {
            "top_k": 10,
            "dense_top_k": 30,
            "sparse_top_k": 30,
            "rerank_top_k": 30,
            "scenario_queries": {"variants": variants},
            "short_ambiguous_queries": {"variants": variants},
            "exact_guardrail_queries": {"variants": variants},
        }
    }
    (project / "experiments" / "stage5_query_eval.json").write_text(__import__("json").dumps(payload), encoding="utf-8")

    path = generate_phase5_report(project)

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "阶段五收口报告" in text
    assert "guardrail" in text
