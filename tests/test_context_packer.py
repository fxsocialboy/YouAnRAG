from __future__ import annotations

from rag_v2.retrieval.context_packer import ContextPacker, _stage4_sort_score
from rag_v2.retrieval.hybrid_searcher import HybridSearchResult


def row(source_file: str, chunk_index: int, section: str, content: str, token_count: int = 10) -> dict:
    return {
        "chunk_id": f"{source_file}::{chunk_index}",
        "source_file": source_file,
        "chunk_index": chunk_index,
        "section_path_text": section,
        "content": content,
        "token_count": token_count,
        "content_hash": f"hash-{source_file}-{chunk_index}",
        "metadata": {"section_path_text": section},
    }


def candidate(
    source_file: str,
    chunk_index: int,
    section: str,
    content: str,
    fusion_score: float,
    *,
    rerank_score: float | None = None,
    mmr_score: float | None = None,
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=f"{source_file}::{chunk_index}",
        content=content,
        source_file=source_file,
        chunk_index=chunk_index,
        section_path_text=section,
        dense_score=0.8,
        sparse_score=2.0,
        fusion_score=fusion_score,
        rank=1,
        metadata={},
        rerank_score=rerank_score,
        mmr_score=mmr_score,
    )


def test_context_packer_merges_adjacent_chunks():
    metadata = [
        row("doc.md", 0, "A", "first", 8),
        row("doc.md", 1, "A", "second", 9),
        row("doc.md", 2, "B", "third", 7),
    ]
    packer = ContextPacker(metadata)
    result = packer.pack(
        [
            candidate("doc.md", 0, "A", "first", 0.09),
            candidate("doc.md", 1, "A", "second", 0.08),
        ],
        token_budget=50,
        same_section_extra=0,
    )
    assert len(result.evidence_chunks) == 1
    chunk = result.evidence_chunks[0]
    assert chunk.chunk_indexes == [0, 1]
    assert "first" in chunk.content and "second" in chunk.content


def test_context_packer_adds_same_section_neighbors():
    metadata = [
        row("doc.md", 0, "A", "first", 8),
        row("doc.md", 1, "A", "second", 9),
        row("doc.md", 2, "A", "third", 7),
    ]
    packer = ContextPacker(metadata)
    result = packer.pack(
        [candidate("doc.md", 1, "A", "second", 0.1)],
        token_budget=50,
        same_section_extra=1,
    )
    assert len(result.evidence_chunks) == 1
    assert result.evidence_chunks[0].chunk_indexes in ([0, 1, 2], [1, 2], [0, 1])
    assert result.total_tokens <= 50


def test_context_packer_token_budget_and_dedup():
    metadata = [
        row("doc1.md", 0, "A", "alpha", 15),
        row("doc2.md", 0, "B", "beta", 15),
    ]
    packer = ContextPacker(metadata)
    result = packer.pack(
        [
            candidate("doc1.md", 0, "A", "alpha", 0.2),
            candidate("doc1.md", 0, "A", "alpha", 0.19),
            candidate("doc2.md", 0, "B", "beta", 0.18),
        ],
        token_budget=20,
        same_section_extra=0,
    )
    assert len(result.evidence_chunks) == 1
    assert result.total_tokens <= 20
    assert result.evidence_chunks[0].citation_id == "[S1]"


def test_context_packer_marks_insufficient_evidence():
    metadata = [row("doc.md", 0, "A", "alpha", 12)]
    packer = ContextPacker(metadata)
    result = packer.pack([], token_budget=20, min_evidence_chunks=1)
    assert result.insufficient_evidence is True
    assert result.evidence_chunks == []


def test_context_packer_citation_ids_are_stable():
    metadata = [
        row("doc1.md", 0, "A", "alpha", 10),
        row("doc2.md", 0, "B", "beta", 10),
    ]
    packer = ContextPacker(metadata)
    result = packer.pack(
        [
            candidate("doc1.md", 0, "A", "alpha", 0.2),
            candidate("doc2.md", 0, "B", "beta", 0.1),
        ],
        token_budget=30,
    )
    assert [item.citation_id for item in result.evidence_chunks] == ["[S1]", "[S2]"]



def test_stage4_sort_score_priority_is_mmr_then_rerank_then_fusion():
    assert _stage4_sort_score(0.1, rerank_score=0.8, mmr_score=0.3) == 0.3
    assert _stage4_sort_score(0.1, rerank_score=0.8, mmr_score=None) == 0.8
    assert _stage4_sort_score(0.1, rerank_score=None, mmr_score=None) == 0.1


def test_context_packer_keeps_stage3_fusion_order_when_stage4_scores_absent():
    metadata = [
        row("doc1.md", 0, "A", "alpha", 10),
        row("doc2.md", 0, "B", "beta", 10),
    ]
    packer = ContextPacker(metadata)
    result = packer.pack(
        [
            candidate("doc1.md", 0, "A", "alpha", 0.1),
            candidate("doc2.md", 0, "B", "beta", 0.2),
        ],
        token_budget=30,
        same_section_extra=0,
    )
    assert [item.source_file for item in result.evidence_chunks] == ["doc2.md", "doc1.md"]
    assert result.evidence_chunks[0].metadata["sort_score"] == 0.2


def test_context_packer_uses_rerank_score_for_stage4_order():
    metadata = [
        row("doc1.md", 0, "A", "alpha", 10),
        row("doc2.md", 0, "B", "beta", 10),
    ]
    packer = ContextPacker(metadata)
    result = packer.pack(
        [
            candidate("doc1.md", 0, "A", "alpha", 0.1, rerank_score=0.9),
            candidate("doc2.md", 0, "B", "beta", 0.2, rerank_score=0.3),
        ],
        token_budget=30,
        same_section_extra=0,
    )
    assert [item.source_file for item in result.evidence_chunks] == ["doc1.md", "doc2.md"]
    assert result.evidence_chunks[0].metadata["rerank_score"] == 0.9
    assert result.evidence_chunks[0].metadata["sort_score"] == 0.9


def test_context_packer_uses_mmr_score_before_rerank_score_for_stage4_order():
    metadata = [
        row("doc1.md", 0, "A", "alpha", 10),
        row("doc2.md", 0, "B", "beta", 10),
    ]
    packer = ContextPacker(metadata)
    result = packer.pack(
        [
            candidate("doc1.md", 0, "A", "alpha", 0.1, rerank_score=0.9, mmr_score=0.2),
            candidate("doc2.md", 0, "B", "beta", 0.2, rerank_score=0.3, mmr_score=0.8),
        ],
        token_budget=30,
        same_section_extra=0,
    )
    assert [item.source_file for item in result.evidence_chunks] == ["doc2.md", "doc1.md"]
    assert result.evidence_chunks[0].metadata["mmr_score"] == 0.8
    assert result.evidence_chunks[0].metadata["sort_score"] == 0.8


def test_context_packer_preserves_stage4_scores_when_merging_adjacent_chunks():
    metadata = [
        row("doc.md", 0, "A", "first", 8),
        row("doc.md", 1, "A", "second", 9),
    ]
    packer = ContextPacker(metadata)
    result = packer.pack(
        [
            candidate("doc.md", 0, "A", "first", 0.1, rerank_score=0.7, mmr_score=0.6),
            candidate("doc.md", 1, "A", "second", 0.2, rerank_score=0.5, mmr_score=0.4),
        ],
        token_budget=50,
        same_section_extra=0,
    )
    assert len(result.evidence_chunks) == 1
    chunk = result.evidence_chunks[0]
    assert chunk.chunk_indexes == [0, 1]
    assert chunk.metadata["rerank_score"] == 0.7
    assert chunk.metadata["mmr_score"] == 0.6
    assert chunk.metadata["sort_score"] == 0.6


def test_context_packer_preserves_seed_dense_confidence_separately_from_mmr():
    metadata = [
        row("doc.md", 0, "A", "first", 10),
        row("doc.md", 1, "A", "second", 10),
    ]
    packer = ContextPacker(metadata)
    seed = candidate("doc.md", 0, "A", "first", 0.03, rerank_score=None, mmr_score=1.0)
    seed.dense_score = 0.42

    result = packer.pack([seed], token_budget=50, same_section_extra=0)

    chunk = result.evidence_chunks[0]
    assert chunk.metadata["mmr_score"] == 1.0
    assert chunk.metadata["dense_score"] == 0.42
