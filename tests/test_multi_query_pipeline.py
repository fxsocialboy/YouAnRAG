from rag_v2.query.hyde import FakeHydeGenerator
from rag_v2.query.models import QueryBranch
from rag_v2.retrieval.hybrid_searcher import HybridSearchResult
from rag_v2.retrieval.multi_query_pipeline import MultiQueryPipeline, MultiQueryPipelineOptions, fuse_branch_results


def make_result(chunk_id: str, *, rank: int, fusion: float = 0.1, rerank: float | None = None):
    source, idx = chunk_id.split("::")
    return HybridSearchResult(
        chunk_id=chunk_id,
        content=f"content {chunk_id}",
        source_file=source,
        chunk_index=int(idx),
        section_path_text="section",
        dense_score=0.5,
        sparse_score=None,
        fusion_score=fusion,
        rank=rank,
        metadata={"token_count": 10},
        rerank_score=rerank,
    )


class RecordingStage4Pipeline:
    def __init__(self):
        self.calls = []

    def search(self, query, *, filters=None, options=None):
        self.calls.append((query, filters, options))
        if "台风" in query and "热带气旋" not in query:
            return [make_result("a.md::1", rank=1, rerank=0.8), make_result("b.md::2", rank=2, rerank=0.7)]
        if "热带气旋" in query:
            return [make_result("b.md::2", rank=1, rerank=0.9), make_result("c.md::3", rank=2, rerank=0.6)]
        return [make_result("fallback.md::9", rank=1)]


class FailingStage4Pipeline:
    def __init__(self):
        self.calls = 0

    def search(self, query, *, filters=None, options=None):
        self.calls += 1
        if self.calls == 1:
            return [make_result("a.md::1", rank=1)]
        raise RuntimeError("branch failed")


def test_fuse_branch_results_merges_duplicate_chunk_and_records_branches():
    raw = QueryBranch(branch="raw", query="台风学校怎么办", weight=1.0)
    expanded = QueryBranch(branch="expanded", query="台风 热带气旋 学校 防御措施", weight=0.7)

    results = fuse_branch_results(
        [
            (raw, [make_result("a.md::1", rank=1), make_result("b.md::2", rank=2, rerank=0.2)]),
            (expanded, [make_result("b.md::2", rank=1, rerank=0.9), make_result("c.md::3", rank=2)]),
        ],
        top_k=3,
    )

    b = next(item for item in results if item.chunk_id == "b.md::2")
    assert b.rerank_score == 0.9
    assert b.metadata["matched_branches"] == ["raw", "expanded"]
    assert set(b.metadata["branch_details"]) == {"raw", "expanded"}
    assert b.metadata["branch_fusion_score"] == b.fusion_score


def test_fuse_branch_results_branch_weight_affects_ranking():
    raw = QueryBranch(branch="raw", query="raw", weight=1.0)
    expanded = QueryBranch(branch="expanded", query="expanded", weight=0.1)

    results = fuse_branch_results(
        [
            (expanded, [make_result("expanded.md::1", rank=1)]),
            (raw, [make_result("raw.md::1", rank=2)]),
        ],
        top_k=2,
    )

    assert results[0].chunk_id == "raw.md::1"


def test_multi_query_pipeline_runs_query_branches_and_fuses():
    stage4 = RecordingStage4Pipeline()
    pipeline = MultiQueryPipeline(stage4)

    output = pipeline.search("台风黄色预警下学校应该怎么做", options=MultiQueryPipelineOptions(top_k=3, enable_reranker=False, enable_mmr=False))

    assert output.query_plan.query_type == "scenario"
    assert [branch.branch for branch in output.query_branches] == ["raw", "expanded"]
    assert len(stage4.calls) == 2
    assert output.results[0].metadata["matched_branches"]
    assert any(item.chunk_id == "b.md::2" for item in output.results)


def test_multi_query_pipeline_exact_query_stays_conservative():
    stage4 = RecordingStage4Pipeline()
    pipeline = MultiQueryPipeline(stage4)

    output = pipeline.search("IV级气象灾害应急响应一般由谁启动？", options=MultiQueryPipelineOptions(top_k=2))

    assert output.query_plan.query_type == "exact_fact"
    assert [branch.branch for branch in output.query_branches] == ["raw"]
    assert len(stage4.calls) == 1


def test_multi_query_pipeline_branch_failure_falls_back_to_successful_branches():
    pipeline = MultiQueryPipeline(FailingStage4Pipeline())

    output = pipeline.search("台风黄色预警下学校应该怎么做", options=MultiQueryPipelineOptions(top_k=3))

    assert [item.chunk_id for item in output.results] == ["a.md::1"]
    assert output.results[0].metadata["matched_branches"] == ["raw"]


def test_multi_query_pipeline_zero_top_k_returns_empty_with_plan():
    pipeline = MultiQueryPipeline(RecordingStage4Pipeline())

    output = pipeline.search("台风黄色预警下学校应该怎么做", options=MultiQueryPipelineOptions(top_k=0))

    assert output.query_plan.query_type == "scenario"
    assert output.query_branches == []
    assert output.results == []


def test_multi_query_pipeline_rule_hyde_branch_is_dense_only_and_masked_in_metadata():
    stage4 = RecordingStage4Pipeline()
    pipeline = MultiQueryPipeline(stage4, hyde_generator=FakeHydeGenerator(content="?? ?? ???? ????"))

    output = pipeline.search("??????????????", options=MultiQueryPipelineOptions(top_k=3, hyde_mode="rule"))

    assert output.hyde_document is not None
    assert output.hyde_document.used is True
    assert "hyde" in [branch.branch for branch in output.query_branches]
    hyde_call = next(call for call in stage4.calls if call[0] == "?? ?? ???? ????")
    assert hyde_call[2].sparse_top_k == 0
    assert hyde_call[2].enable_reranker is False
    assert hyde_call[2].enable_mmr is False
    for result in output.results:
        details = result.metadata.get("branch_details", {})
        if "hyde" in details:
            assert details["hyde"]["query"] == "<hyde_document>"
