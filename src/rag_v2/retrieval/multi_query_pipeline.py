"""Stage5 branch-aware multi-query retrieval pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from rag_v2.query.analyzer import QueryAnalyzer
from rag_v2.query.hyde import DisabledHydeGenerator, HydeDocument, HydeGenerator
from rag_v2.query.models import QueryBranch, QueryPlan
from rag_v2.query.rewriter import QueryRewriter
from rag_v2.retrieval.hybrid_searcher import HybridSearchResult, reciprocal_rank_score


class Stage4Pipeline(Protocol):
    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        options: Any | None = None,
    ) -> list[HybridSearchResult]: ...


@dataclass(slots=True)
class Stage4BranchOptions:
    """Duck-typed options consumed by Stage4 RerankPipeline.search.

    Keeping this local avoids importing rerank_pipeline and its heavier optional
    dependencies when Stage5 query planning is unit-tested with mocks.
    """

    top_k: int = 10
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rerank_top_k: int = 30
    mmr_pre_candidates: int = 20
    mmr_top_k: int = 10
    enable_reranker: bool = True
    enable_mmr: bool = True

    def validate(self) -> None:
        for name in ("top_k", "dense_top_k", "sparse_top_k", "rerank_top_k", "mmr_pre_candidates", "mmr_top_k"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MultiQueryPipelineOptions:
    top_k: int = 10
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rerank_top_k: int = 30
    mmr_pre_candidates: int = 20
    mmr_top_k: int = 10
    enable_reranker: bool | None = None
    enable_mmr: bool | None = None
    branch_rrf_k: int = 60
    branch_candidate_top_k: int = 10
    max_branches: int | None = None
    hyde_mode: str = "disabled"

    def validate(self) -> None:
        for name in (
            "top_k",
            "dense_top_k",
            "sparse_top_k",
            "rerank_top_k",
            "mmr_pre_candidates",
            "mmr_top_k",
            "branch_rrf_k",
            "branch_candidate_top_k",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.branch_rrf_k == 0:
            raise ValueError("branch_rrf_k must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MultiQuerySearchOutput:
    query_plan: QueryPlan
    query_branches: list[QueryBranch]
    results: list[HybridSearchResult]
    hyde_document: HydeDocument | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_plan": self.query_plan.to_dict(),
            "query_branches": [branch.to_dict() for branch in self.query_branches],
            "hyde_document": self.hyde_document.to_dict() if self.hyde_document else None,
            "results": [result.to_dict() for result in self.results],
        }


class MultiQueryPipeline:
    """Run Stage4 retrieval for multiple query branches and fuse results."""

    def __init__(
        self,
        stage4_pipeline: Stage4Pipeline,
        *,
        analyzer: QueryAnalyzer | None = None,
        rewriter: QueryRewriter | None = None,
        hyde_generator: HydeGenerator | None = None,
    ):
        self.stage4_pipeline = stage4_pipeline
        self.analyzer = analyzer or QueryAnalyzer()
        self.rewriter = rewriter or QueryRewriter()
        self.hyde_generator = hyde_generator or DisabledHydeGenerator()

    @classmethod
    def from_stage4_pipeline(
        cls,
        stage4_pipeline: Stage4Pipeline,
        *,
        analyzer: QueryAnalyzer | None = None,
        rewriter: QueryRewriter | None = None,
        hyde_generator: HydeGenerator | None = None,
    ) -> "MultiQueryPipeline":
        return cls(stage4_pipeline, analyzer=analyzer, rewriter=rewriter, hyde_generator=hyde_generator)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        options: MultiQueryPipelineOptions | None = None,
    ) -> MultiQuerySearchOutput:
        options = options or MultiQueryPipelineOptions()
        options.validate()
        if options.top_k <= 0:
            plan = self.analyzer.analyze(query)
            return MultiQuerySearchOutput(query_plan=plan, query_branches=[], results=[], hyde_document=None)

        plan = self.analyzer.analyze(query)
        branches = self.rewriter.rewrite(plan)
        hyde_document = self.hyde_generator.generate(query, plan)
        if hyde_document.used and hyde_document.content:
            weight = plan.retrieval_policy.branch_weights.get("hyde", 0.6)
            branches.append(QueryBranch(branch="hyde", query=hyde_document.content, weight=weight, reason=hyde_document.reason))

        if options.max_branches is not None:
            non_hyde = [branch for branch in branches if branch.branch != "hyde"]
            hyde = [branch for branch in branches if branch.branch == "hyde"]
            branches = non_hyde[: max(1, options.max_branches)] + hyde[:1]
        if not branches:
            return MultiQuerySearchOutput(query_plan=plan, query_branches=[], results=[], hyde_document=hyde_document)

        stage4_options = self._stage4_options(plan, options)
        hyde_options = self._hyde_stage4_options(plan, options)
        branch_hits: list[tuple[QueryBranch, list[HybridSearchResult]]] = []
        for branch in branches:
            try:
                branch_options = hyde_options if branch.branch == "hyde" else stage4_options
                hits = self.stage4_pipeline.search(branch.query, filters=filters, options=branch_options)
            except Exception:
                hits = []
            branch_hits.append((branch, hits))

        fused = fuse_branch_results(
            branch_hits,
            top_k=options.top_k,
            rrf_k=options.branch_rrf_k,
        )
        return MultiQuerySearchOutput(query_plan=plan, query_branches=branches, results=fused, hyde_document=hyde_document)

    def _stage4_options(self, plan: QueryPlan, options: MultiQueryPipelineOptions) -> Stage4BranchOptions:
        enable_reranker = plan.retrieval_policy.use_reranker if options.enable_reranker is None else options.enable_reranker
        enable_mmr = plan.retrieval_policy.use_mmr if options.enable_mmr is None else options.enable_mmr
        candidate_top_k = max(options.branch_candidate_top_k, options.top_k)
        return Stage4BranchOptions(
            top_k=candidate_top_k,
            dense_top_k=options.dense_top_k or plan.retrieval_policy.dense_top_k,
            sparse_top_k=options.sparse_top_k or plan.retrieval_policy.sparse_top_k,
            rerank_top_k=options.rerank_top_k or plan.retrieval_policy.rerank_top_k,
            mmr_pre_candidates=options.mmr_pre_candidates,
            mmr_top_k=max(options.mmr_top_k, candidate_top_k),
            enable_reranker=enable_reranker,
            enable_mmr=enable_mmr,
        )

    def _hyde_stage4_options(self, plan: QueryPlan, options: MultiQueryPipelineOptions) -> Stage4BranchOptions:
        candidate_top_k = max(options.branch_candidate_top_k, options.top_k)
        return Stage4BranchOptions(
            top_k=candidate_top_k,
            dense_top_k=options.dense_top_k or plan.retrieval_policy.dense_top_k,
            sparse_top_k=0,
            rerank_top_k=0,
            mmr_pre_candidates=0,
            mmr_top_k=candidate_top_k,
            enable_reranker=False,
            enable_mmr=False,
        )


def fuse_branch_results(
    branch_hits: list[tuple[QueryBranch, list[HybridSearchResult]]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[HybridSearchResult]:
    """Weighted branch-aware RRF over Stage4 branch results."""

    if top_k <= 0:
        return []
    fused: dict[str, dict[str, Any]] = {}

    for branch, hits in branch_hits:
        for branch_rank, hit in enumerate(hits, 1):
            entry = fused.setdefault(
                hit.chunk_id,
                {
                    "result": clone_result(hit),
                    "branch_score": 0.0,
                    "matched_branches": [],
                    "branch_details": {},
                    "best_rank": branch_rank,
                },
            )
            score = float(branch.weight) * reciprocal_rank_score(branch_rank, rrf_k)
            entry["branch_score"] += score
            entry["best_rank"] = min(int(entry["best_rank"]), branch_rank)
            if branch.branch not in entry["matched_branches"]:
                entry["matched_branches"].append(branch.branch)
            entry["branch_details"][branch.branch] = {
                "query": "<hyde_document>" if branch.branch == "hyde" else branch.query,
                "weight": branch.weight,
                "rank": branch_rank,
                "branch_score": round(score, 6),
                "fusion_score": hit.fusion_score,
                "rerank_score": hit.rerank_score,
                "mmr_score": hit.mmr_score,
            }
            merge_best_scores(entry["result"], hit)

    ranked = sorted(
        fused.values(),
        key=lambda item: (-float(item["branch_score"]), int(item["best_rank"]), item["result"].chunk_id),
    )
    selected: list[HybridSearchResult] = []
    for rank, entry in enumerate(ranked[:top_k], 1):
        result: HybridSearchResult = entry["result"]
        result.rank = rank
        result.stage4_rank = rank if result.rerank_score is not None or result.mmr_score is not None else result.stage4_rank
        result.fusion_score = round(float(entry["branch_score"]), 6)
        result.metadata = dict(result.metadata)
        result.metadata["matched_branches"] = list(entry["matched_branches"])
        result.metadata["branch_details"] = dict(entry["branch_details"])
        result.metadata["branch_fusion_score"] = result.fusion_score
        selected.append(result)
    return selected


def clone_result(result: HybridSearchResult) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=result.chunk_id,
        content=result.content,
        source_file=result.source_file,
        chunk_index=result.chunk_index,
        section_path_text=result.section_path_text,
        dense_score=result.dense_score,
        sparse_score=result.sparse_score,
        fusion_score=result.fusion_score,
        rank=result.rank,
        metadata=dict(result.metadata),
        rerank_score=result.rerank_score,
        mmr_score=result.mmr_score,
        stage4_rank=result.stage4_rank,
    )


def merge_best_scores(target: HybridSearchResult, candidate: HybridSearchResult) -> None:
    if candidate.dense_score is not None and (target.dense_score is None or candidate.dense_score > target.dense_score):
        target.dense_score = candidate.dense_score
    if candidate.sparse_score is not None and (target.sparse_score is None or candidate.sparse_score > target.sparse_score):
        target.sparse_score = candidate.sparse_score
    if candidate.rerank_score is not None and (target.rerank_score is None or candidate.rerank_score > target.rerank_score):
        target.rerank_score = candidate.rerank_score
    if candidate.mmr_score is not None and (target.mmr_score is None or candidate.mmr_score > target.mmr_score):
        target.mmr_score = candidate.mmr_score
