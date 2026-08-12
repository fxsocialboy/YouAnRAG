import json
from pathlib import Path

import numpy as np

import evaluate_stage1 as ev
from rag_v2.retrieval.stage1_searcher import Stage1Searcher
from rag_v2.stores.faiss_store import FaissStore


class FakeEmbedder:
    def encode_queries(self, queries):
        if "台风" in queries[0]:
            return np.array([[1.0, 0.0]], dtype="float32")
        return np.array([[0.0, 1.0]], dtype="float32")


def make_fake_searcher():
    metadata = [
        {
            "chunk_id": "a.md::0",
            "source_file": "a.md",
            "chunk_index": 0,
            "section_path": ["a"],
            "section_path_text": "a",
            "content": "台风材料",
            "token_count": 10,
        },
        {
            "chunk_id": "b.md::0",
            "source_file": "b.md",
            "chunk_index": 0,
            "section_path": ["b"],
            "section_path_text": "b",
            "content": "地震材料",
            "token_count": 10,
        },
    ]
    store = FaissStore.from_vectors(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"), metadata)
    return Stage1Searcher(store=store, embedder=FakeEmbedder())


def test_evaluate_stage1_with_fake_searcher(monkeypatch):
    base = Path(__file__).resolve().parents[1] / "artifacts" / "stage1" / "test_tmp_eval_stage1"
    base.mkdir(parents=True, exist_ok=True)
    queries = base / "queries.jsonl"
    out = base / "out.json"
    try:
        queries.write_text(
            '\n'.join(
                [
                    json.dumps({"id": "q1", "query": "台风怎么办", "relevant_source_file": "a.md", "relevant_chunk_index": 0}, ensure_ascii=False),
                    json.dumps({"id": "q2", "query": "地震怎么办", "relevant_source_file": "b.md", "relevant_chunk_index": 0}, ensure_ascii=False),
                ]
            )
            + '\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(ev.Stage1Searcher, "from_config", classmethod(lambda cls, device="cpu", batch_size=16: make_fake_searcher()))
        result = ev.evaluate_stage1(queries, out, top_k=1)
        assert result["summary"]["doc_recall@10"] == 1.0
        assert result["summary"]["chunk_recall_exact_index@10"] == 1.0
        assert out.exists()
    finally:
        for p in [queries, out]:
            if p.exists():
                p.unlink()


def test_write_comparison_report():
    base = Path(__file__).resolve().parents[1] / "artifacts" / "stage1" / "test_tmp_eval_stage1"
    base.mkdir(parents=True, exist_ok=True)
    legacy = base / "legacy.json"
    report = base / "report.md"
    try:
        legacy.write_text(json.dumps({"summary": {"doc_recall@10": 0.5, "chunk_recall@10": 0.2}}, ensure_ascii=False), encoding="utf-8")
        ev.write_comparison_report({"summary": {"doc_recall@10": 0.8, "chunk_recall_exact_index@10": 0.1}}, legacy, report)
        text = report.read_text(encoding="utf-8")
        assert "Stage1 vs Legacy" in text
        assert "doc_recall@10" in text
    finally:
        for p in [legacy, report]:
            if p.exists():
                p.unlink()
