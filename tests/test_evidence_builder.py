from rag_v2.agent.evidence import EvidenceBuilder
from rag_v2.retrieval.hybrid_searcher import HybridSearchResult
from rag_v2.retrieval.context_packer import ContextChunk


def make_result(rank, chunk_id, score, content="content", source_file="a.md"):
    return HybridSearchResult(
        chunk_id=chunk_id,
        content=content,
        source_file=source_file,
        chunk_index=rank,
        section_path_text="一、响应措施",
        dense_score=score,
        sparse_score=None,
        fusion_score=score,
        rank=rank,
        metadata={"matched_branches": ["raw"]},
    )


def test_evidence_builder_sorts_by_score_and_numbers_by_rank():
    evidence = EvidenceBuilder().build_from_search_results(
        [
            make_result(1, "a.md::1", 0.2),
            make_result(2, "a.md::2", 0.9),
        ]
    )

    assert [item.chunk_id for item in evidence] == ["a.md::2", "a.md::1"]
    assert [item.citation_id for item in evidence] == ["S1", "S2"]
    assert evidence[0].marker == "[S1]"


def test_evidence_builder_deduplicates_same_chunk_and_keeps_best_score():
    evidence = EvidenceBuilder().build_from_search_results(
        [
            make_result(1, "a.md::1", 0.2, content="old"),
            make_result(2, "a.md::1", 0.8, content="best"),
        ]
    )

    assert len(evidence) == 1
    assert evidence[0].content == "best"
    assert evidence[0].score == 0.8
    assert evidence[0].citation_id == "S1"


def test_evidence_builder_limits_max_items():
    evidence = EvidenceBuilder(max_items=1).build_from_search_results(
        [make_result(1, "a.md::1", 0.2), make_result(2, "a.md::2", 0.9)]
    )

    assert len(evidence) == 1
    assert evidence[0].chunk_id == "a.md::2"


def test_context_evidence_uses_rerank_score_for_guardrail_confidence():
    chunk = ContextChunk(
        citation_id="S1",
        source_file="a.md",
        section_path_text="响应措施",
        chunk_ids=["a.md::1"],
        chunk_indexes=[1],
        content="应立即组织疏散。",
        token_count=10,
        fusion_score=0.03,
        metadata={"rerank_score": 0.91, "mmr_score": 0.72},
    )

    evidence = EvidenceBuilder().build_from_context_chunks([chunk])

    assert evidence[0].score == 0.91
    assert evidence[0].metadata["fusion_score"] == 0.03


def test_mmr_score_never_overwrites_rerank_retrieval_confidence():
    chunk = ContextChunk(
        citation_id="S1",
        source_file="a.md",
        section_path_text="响应措施",
        chunk_ids=["a.md::1"],
        chunk_indexes=[1],
        content="措施",
        token_count=10,
        fusion_score=0.03,
        metadata={"rerank_score": 0.12, "mmr_score": 1.0, "sort_score": 1.0},
    )

    evidence = EvidenceBuilder().build_from_context_chunks([chunk])

    assert evidence[0].score == 0.12
    assert evidence[0].metadata["mmr_score"] == 1.0
    assert evidence[0].metadata["confidence_type"] == "rerank"
    assert evidence[0].metadata["confidence_state"] == "low_confidence"


def test_rrf_only_evidence_marks_confidence_unavailable():
    result = make_result(1, "a.md::1", 0.03)
    result.dense_score = None
    evidence = EvidenceBuilder().build_from_search_results([result])

    assert evidence[0].score == 0.0
    assert evidence[0].metadata["fusion_score"] == 0.03
    assert evidence[0].metadata["confidence_type"] == "unavailable"
