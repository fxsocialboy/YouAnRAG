from __future__ import annotations

from rag_v2.retrieval.bm25_index import BM25SearchHit
from rag_v2.retrieval.hybrid_searcher import HybridSearcher, HybridSearchResult, reciprocal_rank_score
from rag_v2.retrieval.qdrant_searcher import Stage2SearchResult


class FakeDenseSearcher:
    def __init__(self, hits: list[Stage2SearchResult]):
        self.hits = hits
        self.calls = []

    def search(self, query: str, top_k: int = 10, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return self.hits[:top_k]


class FakeSparseIndex:
    def __init__(self, hits: list[BM25SearchHit]):
        self.hits = hits
        self.calls = []

    def search(self, query: str, top_k: int = 10, source_file: str | None = None):
        self.calls.append({"query": query, "top_k": top_k, "source_file": source_file})
        if source_file:
            return [hit for hit in self.hits if hit.source_file == source_file][:top_k]
        return self.hits[:top_k]


def dense_hit(chunk_id: str, rank: int, score: float, source_file: str = "doc.md", chunk_index: int = 0):
    return Stage2SearchResult(
        rank=rank,
        score=score,
        source_file=source_file,
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        section_path=[],
        section_path_text="Section",
        content=f"{chunk_id} dense content",
        content_preview=f"{chunk_id} dense preview",
        token_count=10,
    )


def sparse_hit(chunk_id: str, rank: int, score: float, source_file: str = "doc.md", chunk_index: int = 0):
    return BM25SearchHit(
        rank=rank,
        score=score,
        chunk_id=chunk_id,
        source_file=source_file,
        chunk_index=chunk_index,
        section_path_text="Section",
        content=f"{chunk_id} sparse content",
        content_preview=f"{chunk_id} sparse preview",
        token_count=8,
        matched_terms=["iv", "响应"],
    )


def test_hybrid_dense_only_and_sparse_only_hits_are_kept():
    dense = FakeDenseSearcher([dense_hit("a", 1, 0.9), dense_hit("b", 2, 0.8)])
    sparse = FakeSparseIndex([sparse_hit("c", 1, 3.2)])
    searcher = HybridSearcher(dense_searcher=dense, sparse_index=sparse, rrf_k=60)

    results = searcher.search("query", top_k=3, dense_top_k=3, sparse_top_k=3)
    chunk_ids = [item.chunk_id for item in results]
    assert set(chunk_ids) == {"a", "b", "c"}


def test_hybrid_rrf_fusion_scores_are_correct_for_overlap():
    dense = FakeDenseSearcher([dense_hit("shared", 1, 0.95)])
    sparse = FakeSparseIndex([sparse_hit("shared", 2, 4.2)])
    searcher = HybridSearcher(dense_searcher=dense, sparse_index=sparse, rrf_k=60)

    result = searcher.search("query", top_k=1)[0]
    expected = reciprocal_rank_score(1, 60) + reciprocal_rank_score(2, 60)
    assert result.chunk_id == "shared"
    assert result.dense_score == 0.95
    assert result.sparse_score == 4.2
    assert round(result.fusion_score, 6) == round(expected, 6)


def test_hybrid_deduplicates_chunk_ids_and_passes_source_file_filter():
    dense = FakeDenseSearcher([dense_hit("shared", 1, 0.9, source_file="target.md")])
    sparse = FakeSparseIndex(
        [
            sparse_hit("shared", 1, 3.5, source_file="target.md"),
            sparse_hit("other", 2, 2.0, source_file="other.md"),
        ]
    )
    searcher = HybridSearcher(dense_searcher=dense, sparse_index=sparse)

    results = searcher.search("query", top_k=5, filters={"source_file": "target.md"})
    assert [item.chunk_id for item in results] == ["shared"]
    assert dense.calls[0]["filters"] == {"source_file": "target.md"}
    assert sparse.calls[0]["source_file"] == "target.md"


def test_hybrid_invoke_returns_text_list():
    dense = FakeDenseSearcher([dense_hit("a", 1, 0.9)])
    sparse = FakeSparseIndex([sparse_hit("b", 1, 3.0)])
    searcher = HybridSearcher(dense_searcher=dense, sparse_index=sparse)

    contents = searcher.invoke("query", top_k=2)
    assert len(contents) == 2
    assert any("dense content" in text or "sparse content" in text for text in contents)



def test_hybrid_result_stage4_fields_are_optional_and_serialized():
    result = HybridSearchResult(
        chunk_id="stage4.md::0",
        content="content",
        source_file="stage4.md",
        chunk_index=0,
        section_path_text="Section",
        dense_score=0.8,
        sparse_score=None,
        fusion_score=0.016,
        rank=1,
        metadata={},
    )

    payload = result.to_dict()
    assert result.rerank_score is None
    assert result.mmr_score is None
    assert result.stage4_rank is None
    assert payload["rerank_score"] is None
    assert payload["mmr_score"] is None
    assert payload["stage4_rank"] is None


def test_hybrid_search_outputs_stage4_fields_without_changing_rrf_order():
    dense = FakeDenseSearcher([dense_hit("a", 1, 0.9), dense_hit("b", 2, 0.8)])
    sparse = FakeSparseIndex([])
    searcher = HybridSearcher(dense_searcher=dense, sparse_index=sparse, rrf_k=60)

    results = searcher.search("query", top_k=2)
    assert [item.chunk_id for item in results] == ["a", "b"]
    assert all(item.rerank_score is None for item in results)
    assert all(item.mmr_score is None for item in results)
    assert all(item.stage4_rank is None for item in results)
    assert "rerank_score" in results[0].to_dict()
