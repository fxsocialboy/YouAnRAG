import numpy as np
from pathlib import Path

from rag_v2.retrieval.stage1_searcher import Stage1Searcher, load_metadata
from rag_v2.stores.faiss_store import FaissStore


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = np.asarray([vector], dtype="float32")

    def encode_queries(self, queries):
        assert queries
        return self.vector


def metadata_rows():
    return [
        {
            "chunk_id": "a.md::0",
            "source_file": "a.md",
            "chunk_index": 0,
            "section_path": ["a", "sec"],
            "section_path_text": "a > sec",
            "content": "台风黄色预警下学校应该停课并组织避险。",
            "token_count": 20,
        },
        {
            "chunk_id": "b.md::0",
            "source_file": "b.md",
            "chunk_index": 0,
            "section_path": ["b"],
            "section_path_text": "b",
            "content": "地震后应疏散居民。",
            "token_count": 12,
        },
    ]


def test_stage1_searcher_returns_structured_results():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store = FaissStore.from_vectors(vectors, metadata_rows())
    searcher = Stage1Searcher(store=store, embedder=FakeEmbedder([1.0, 0.0]))
    results = searcher.search("台风黄色预警下学校应该怎么做", top_k=2)
    assert len(results) == 2
    assert results[0].source_file == "a.md"
    assert results[0].chunk_index == 0
    assert results[0].section_path_text == "a > sec"
    assert "台风黄色预警" in results[0].content_preview
    assert results[0].to_dict()["rank"] == 1


def test_stage1_searcher_search_dicts():
    store = FaissStore.from_vectors(np.array([[1.0, 0.0]], dtype="float32"), metadata_rows()[:1])
    searcher = Stage1Searcher(store=store, embedder=FakeEmbedder([1.0, 0.0]))
    rows = searcher.search_dicts("query", top_k=1)
    assert rows[0]["source_file"] == "a.md"
    assert rows[0]["content_preview"]


def test_load_metadata():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "stage1" / "test_tmp_stage1_searcher_metadata.json"
    try:
        path.write_text('[{"source_file":"a.md"}]', encoding="utf-8")
        assert load_metadata(path) == [{"source_file": "a.md"}]
    finally:
        if path.exists():
            path.unlink()

