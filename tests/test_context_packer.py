from __future__ import annotations

from rag_v2.retrieval.context_packer import ContextPacker
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


def candidate(source_file: str, chunk_index: int, section: str, content: str, fusion_score: float) -> HybridSearchResult:
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
