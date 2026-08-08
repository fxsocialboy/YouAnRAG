import json
from pathlib import Path
from types import SimpleNamespace

import evaluate_stage3 as ev
from rag_v2.retrieval.bm25_index import BM25SearchHit
from rag_v2.retrieval.context_packer import ContextPacker
from rag_v2.retrieval.hybrid_searcher import HybridSearcher
from rag_v2.retrieval.qdrant_searcher import Stage2SearchResult
import scripts.generate_phase3_report as report_mod


class FakeDense:
    def __init__(self, mapping):
        self.mapping = mapping
        self.closed = False

    def search(self, query, top_k=10, filters=None):
        items = list(self.mapping.get(query, []))
        if filters and filters.get("source_file"):
            items = [item for item in items if item.source_file == filters["source_file"]]
        return items[:top_k]

    def close(self):
        self.closed = True


class FakeSparse:
    def __init__(self, mapping):
        self.mapping = mapping

    def search(self, query, top_k=10, source_file=None):
        items = list(self.mapping.get(query, []))
        if source_file:
            items = [item for item in items if item.source_file == source_file]
        return items[:top_k]


class FakeBM25Loader:
    @staticmethod
    def load(_path):
        return SPARSE_INDEX


DENSE_MAPPING = {
    "bm25 keyword": [
        Stage2SearchResult(
            rank=1,
            score=0.91,
            source_file="other.md",
            chunk_index=0,
            chunk_id="other.md::0",
            section_path=["Other"],
            section_path_text="Other",
            content="其他文档内容",
            content_preview="其他文档内容",
            token_count=12,
        )
    ],
    "自然语言问题": [
        Stage2SearchResult(
            rank=1,
            score=0.95,
            source_file="doc_a.md",
            chunk_index=0,
            chunk_id="doc_a.md::0",
            section_path=["总则"],
            section_path_text="总则",
            content="自然语言问题相关答案",
            content_preview="自然语言问题相关答案",
            token_count=18,
        )
    ],
}

SPARSE_MAPPING = {
    "bm25 keyword": [
        BM25SearchHit(
            rank=1,
            score=7.2,
            chunk_id="doc_b.md::0",
            source_file="doc_b.md",
            chunk_index=0,
            section_path_text="附则",
            content="BM25 关键词命中文档",
            content_preview="BM25 关键词命中文档",
            token_count=16,
            matched_terms=["bm25", "keyword"],
        )
    ],
    "自然语言问题": [
        BM25SearchHit(
            rank=1,
            score=5.0,
            chunk_id="doc_a.md::0",
            source_file="doc_a.md",
            chunk_index=0,
            section_path_text="总则",
            content="自然语言问题相关答案",
            content_preview="自然语言问题相关答案",
            token_count=18,
            matched_terms=["自然语言", "问题"],
        )
    ],
}

SPARSE_INDEX = FakeSparse(SPARSE_MAPPING)

METADATA_ROWS = [
    {
        "chunk_id": "doc_a.md::0",
        "source_file": "doc_a.md",
        "chunk_index": 0,
        "section_path": ["总则"],
        "section_path_text": "总则",
        "content": "自然语言问题相关答案",
        "token_count": 18,
        "content_hash": "ha",
    },
    {
        "chunk_id": "doc_b.md::0",
        "source_file": "doc_b.md",
        "chunk_index": 0,
        "section_path": ["附则"],
        "section_path_text": "附则",
        "content": "BM25 关键词命中文档",
        "token_count": 16,
        "content_hash": "hb",
    },
    {
        "chunk_id": "other.md::0",
        "source_file": "other.md",
        "chunk_index": 0,
        "section_path": ["Other"],
        "section_path_text": "Other",
        "content": "其他文档内容",
        "token_count": 12,
        "content_hash": "hc",
    },
]


def test_evaluate_queries_hybrid_improves_keyword_case():
    dense = FakeDense(DENSE_MAPPING)
    hybrid = HybridSearcher(dense_searcher=dense, sparse_index=SPARSE_INDEX)
    packer = ContextPacker(METADATA_ROWS)
    queries = [
        {"id": "q1", "query": "bm25 keyword", "relevant_source_file": "doc_b.md", "relevant_chunk_index": 0},
        {"id": "q2", "query": "自然语言问题", "relevant_source_file": "doc_a.md", "relevant_chunk_index": 0},
    ]
    summary, rows = ev.evaluate_queries(
        queries,
        dense=dense,
        sparse=SPARSE_INDEX,
        hybrid=hybrid,
        packer=packer,
        source_files=["doc_a.md", "doc_b.md", "other.md"],
        top_k=10,
        dense_top_k=5,
        sparse_top_k=5,
        token_budget=100,
    )
    assert summary["doc_recall@10"]["dense"] == 0.5
    assert summary["doc_recall@10"]["hybrid"] == 1.0
    assert summary["packed_nonempty_ratio"] == 1.0
    assert rows[0]["hybrid_doc_hit@10"] is True


def test_generate_phase3_report_writes_markdown(monkeypatch):
    base = Path(r"G:\tiaozhanbei\newrag\artifacts\stage3\test_tmp_report")
    experiments_dir = base / "experiments"
    artifacts_dir = base / "artifacts"
    stage3_dir = artifacts_dir / "stage3"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    stage3_dir.mkdir(parents=True, exist_ok=True)

    evaluation = {
        "summary": {
            "natural_queries": {
                "doc_recall@10": {"dense": 0.8, "sparse": 0.6, "hybrid": 0.9},
                "chunk_recall_exact@10": {"dense": 0.5, "sparse": 0.4, "hybrid": 0.7},
                "hybrid_minus_dense_doc_recall@10": 0.1,
                "filter_pass_ratio": 1.0,
                "packed_nonempty_ratio": 1.0,
                "avg_packed_token_ratio": 0.6,
                "avg_latency_ms": 12.3,
            },
            "keyword_queries": {
                "doc_recall@10": {"dense": 0.4, "sparse": 0.8, "hybrid": 0.9},
                "chunk_recall_exact@10": {"dense": 0.2, "sparse": 0.7, "hybrid": 0.8},
                "hybrid_minus_dense_doc_recall@10": 0.5,
                "filter_pass_ratio": 1.0,
                "packed_nonempty_ratio": 1.0,
                "avg_packed_token_ratio": 0.55,
                "avg_latency_ms": 10.1,
            },
        },
        "natural_results": [],
        "keyword_results": [
            {
                "id": "k1",
                "query": "bm25 keyword",
                "dense_doc_hit@10": False,
                "hybrid_doc_hit@10": True,
                "hybrid_top10": [{"source_file": "doc_b.md"}],
            }
        ],
    }
    (experiments_dir / "stage3_hybrid_eval.json").write_text(json.dumps(evaluation, ensure_ascii=False), encoding="utf-8")

    fake_cfg = SimpleNamespace(artifacts_dir=artifacts_dir)
    monkeypatch.setattr(report_mod, "get_config", lambda: fake_cfg)
    out = report_mod.generate_phase3_report(project_root=base)
    text = out.read_text(encoding="utf-8")
    assert out.exists()
    assert "阶段三收口报告" in text
    assert "dense + sparse + context packing" in text
    assert "k1" in text
