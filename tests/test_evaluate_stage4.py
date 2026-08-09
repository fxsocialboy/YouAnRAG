import json
from pathlib import Path
from types import SimpleNamespace

import evaluate_stage4 as ev
import scripts.generate_phase4_report as report_mod
from rag_v2.retrieval.context_packer import ContextPacker
from rag_v2.retrieval.hybrid_searcher import HybridSearchResult
from rag_v2.retrieval.rerank_pipeline import RerankPipeline
from rag_v2.retrieval.reranker import FakeReranker


def candidate(chunk_id: str, source_file: str, chunk_index: int, rank: int, fusion_score: float) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        content=f"{chunk_id} content",
        source_file=source_file,
        chunk_index=chunk_index,
        section_path_text="S",
        dense_score=None,
        sparse_score=None,
        fusion_score=fusion_score,
        rank=rank,
        metadata={},
    )


class FakeHybrid:
    def search(self, query, *, top_k=10, dense_top_k=30, sparse_top_k=30, filters=None):
        return [
            candidate("wrong.md::0", "wrong.md", 0, 1, 0.03),
            candidate("right.md::0", "right.md", 0, 2, 0.02),
            candidate("other.md::0", "other.md", 0, 3, 0.01),
        ][:top_k]


class FakeMMR:
    def select(self, candidates, top_k):
        selected = candidates[:top_k]
        for rank, item in enumerate(selected, 1):
            item.mmr_score = 1.0 / rank
            item.rank = rank
            item.stage4_rank = rank
        return selected


METADATA = [
    {"chunk_id": "wrong.md::0", "source_file": "wrong.md", "chunk_index": 0, "section_path_text": "S", "content": "wrong", "token_count": 5, "content_hash": "hw"},
    {"chunk_id": "right.md::0", "source_file": "right.md", "chunk_index": 0, "section_path_text": "S", "content": "right", "token_count": 5, "content_hash": "hr"},
    {"chunk_id": "other.md::0", "source_file": "other.md", "chunk_index": 0, "section_path_text": "S", "content": "other", "token_count": 5, "content_hash": "ho"},
]


def test_rank_helpers_and_shift():
    items = [{"source_file": "a.md", "chunk_index": 0}, {"source_file": "b.md", "chunk_index": 3}]
    assert ev.rank_of_doc(items, "b.md", 10) == 2
    assert ev.rank_of_doc(items, "x.md", 10) is None
    assert ev.rank_of_chunk(items, "b.md", 3, 10) == 2
    assert ev.reciprocal_rank(2) == 0.5
    assert ev.rank_shift(3, 1, 11) == -2
    assert ev.rank_shift(None, 2, 11) == -9
    assert ev.rank_shift(None, None, 11) is None


def test_evaluate_query_set_computes_four_variant_metrics():
    pipeline = RerankPipeline(
        FakeHybrid(),
        reranker=FakeReranker({"right.md::0": 1.0, "wrong.md::0": 0.1, "other.md::0": 0.0}),
        mmr_selector=FakeMMR(),
    )
    packer = ContextPacker(METADATA)
    queries = [{"id": "q1", "query": "test", "relevant_source_file": "right.md", "relevant_chunk_index": 0}]

    summary, rows = ev.evaluate_query_set(
        queries,
        source_files=["right.md", "wrong.md", "other.md"],
        pipeline=pipeline,
        packer=packer,
        top_k=3,
        dense_top_k=3,
        sparse_top_k=3,
        rerank_top_k=3,
        mmr_pre_candidates=3,
        mmr_top_k=3,
        token_budget=100,
    )

    assert set(summary["variants"]) == set(ev.VARIANTS)
    assert summary["variants"]["hybrid"]["doc_mrr@10"] == 0.5
    assert summary["variants"]["hybrid_rerank"]["doc_mrr@10"] == 1.0
    assert rows[0]["variant_metrics"]["hybrid_rerank"]["doc_rank_shift_vs_hybrid"] == -1
    assert rows[0]["variant_results"]["hybrid_rerank"][0]["source_file"] == "right.md"


def test_summarize_rows_adds_deltas():
    rows = [
        {
            "variant_metrics": {
                "hybrid": {"doc_hit@10": True, "chunk_hit_exact@10": False, "doc_mrr@10": 0.5, "chunk_mrr_exact@10": 0, "latency_ms": 1, "packed_nonempty": True, "packed_token_ratio": 0.1, "doc_rank_shift_vs_hybrid": 0, "chunk_rank_shift_vs_hybrid": None},
                "hybrid_rerank": {"doc_hit@10": True, "chunk_hit_exact@10": True, "doc_mrr@10": 1, "chunk_mrr_exact@10": 1, "latency_ms": 2, "packed_nonempty": True, "packed_token_ratio": 0.2, "doc_rank_shift_vs_hybrid": -1, "chunk_rank_shift_vs_hybrid": -1},
                "hybrid_mmr": {"doc_hit@10": True, "chunk_hit_exact@10": False, "doc_mrr@10": 0.5, "chunk_mrr_exact@10": 0, "latency_ms": 3, "packed_nonempty": True, "packed_token_ratio": 0.3, "doc_rank_shift_vs_hybrid": 0, "chunk_rank_shift_vs_hybrid": None},
                "hybrid_rerank_mmr": {"doc_hit@10": True, "chunk_hit_exact@10": True, "doc_mrr@10": 1, "chunk_mrr_exact@10": 1, "latency_ms": 4, "packed_nonempty": True, "packed_token_ratio": 0.4, "doc_rank_shift_vs_hybrid": -1, "chunk_rank_shift_vs_hybrid": -1},
            }
        }
    ]
    summary = ev.summarize_rows(rows, top_k=10)
    assert summary["variants"]["hybrid_rerank"]["doc_mrr_delta_vs_hybrid"] == 0.5
    assert summary["variants"]["hybrid_rerank_mmr"]["avg_doc_rank_shift_vs_hybrid"] == -1.0


def test_generate_phase4_report_writes_markdown(monkeypatch):
    base = Path(r"G:\tiaozhanbei\newrag\artifacts\stage4\test_tmp_report")
    experiments_dir = base / "experiments"
    artifacts_dir = base / "artifacts"
    stage4_dir = artifacts_dir / "stage4"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    stage4_dir.mkdir(parents=True, exist_ok=True)
    variant_metrics = {
        name: {
            "doc_recall@10": 1.0,
            "chunk_recall_exact@10": 0.5,
            "doc_mrr@10": 0.8,
            "chunk_mrr_exact@10": 0.4,
            "doc_mrr_delta_vs_hybrid": 0.1,
            "avg_doc_rank_shift_vs_hybrid": -1.0,
            "avg_latency_ms": 10.0,
        }
        for name in ev.VARIANTS
    }
    evaluation = {
        "summary": {
            "top_k": 10,
            "dense_top_k": 30,
            "sparse_top_k": 30,
            "rerank_top_k": 30,
            "mmr_pre_candidates": 20,
            "mmr_top_k": 10,
            "fake_reranker": True,
            "reranker_model_ref": "fake",
            "natural_queries": {"variants": variant_metrics},
            "keyword_queries": {"variants": variant_metrics},
        },
        "natural_results": [
            {"id": "q1", "query": "test", "variant_metrics": {"hybrid_rerank_mmr": {"doc_rank_shift_vs_hybrid": -1, "chunk_rank_shift_vs_hybrid": -1}}}
        ],
        "keyword_results": [],
    }
    (experiments_dir / "stage4_rerank_eval.json").write_text(json.dumps(evaluation, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(report_mod, "get_config", lambda: SimpleNamespace(stage4_artifacts_dir=stage4_dir))

    out = report_mod.generate_phase4_report(project_root=base)
    text = out.read_text(encoding="utf-8")

    assert out.exists()
    assert "阶段四收口报告" in text
    assert "hybrid + rerank + MMR" in text
    assert "MRR@10" in text
