import pytest

from rag_v2.evaluation.retrieval_metrics import (
    evaluate_ranked_results,
    ndcg_at_k,
    reciprocal_rank_at_k,
    recall_at_k,
    summarize_retrieval_rows,
)


def test_recall_mrr_and_ndcg_for_known_ranking():
    ranked = ["x", "a", "b", "c"]
    relevant = ["a", "c"]

    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == 1.0
    assert reciprocal_rank_at_k(ranked, relevant, 4) == 0.5
    assert ndcg_at_k(ranked, relevant, 4) == pytest.approx(
        (1 / 1.584962500721156 + 1 / 2.321928094887362) / (1 + 1 / 1.584962500721156)
    )


def test_metrics_return_none_without_relevance_labels():
    assert recall_at_k(["a"], [], 10) is None
    assert reciprocal_rank_at_k(["a"], [], 10) is None
    assert ndcg_at_k(["a"], [], 10) is None


def test_evaluate_ranked_results_separates_doc_and_chunk_metrics():
    results = [
        {"source_file": "noise.md", "chunk_id": "noise.md::0"},
        {"source_file": "right.md", "chunk_id": "right.md::2"},
    ]
    metrics = evaluate_ranked_results(
        results,
        relevant_source_files=["right.md"],
        relevant_chunk_ids=["right.md::2"],
        ks=(1, 2),
    )

    assert metrics["doc_recall@1"] == 0.0
    assert metrics["chunk_recall@2"] == 1.0
    assert metrics["doc_mrr@2"] == 0.5


def test_summarize_retrieval_rows_ignores_unlabeled_none_values():
    summary = summarize_retrieval_rows(
        [
            {"doc_recall@10": 1.0, "latency_ms": 10},
            {"doc_recall@10": None, "latency_ms": 30},
        ]
    )

    assert summary["query_count"] == 2
    assert summary["doc_recall@10"] == 1.0
    assert summary["avg_latency_ms"] == 20.0
    assert summary["p50_latency_ms"] == 20.0
