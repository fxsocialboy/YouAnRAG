"""Stage6 Agent-facing answer service.

This module turns the Stage5 retrieval pipeline into a reusable Python API for
CLI/FastAPI/Agent integration.  Heavy retrieval dependencies are imported only
inside ``from_config`` so unit tests and downstream Agent code can import the
service without loading BGE, Qdrant or Cross-Encoder models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Protocol

from rag_v2.agent.composer import TemplateAnswerComposer
from rag_v2.agent.evidence import EvidenceBuilder
from rag_v2.agent.domain_gate import DeepSeekDomainClassifier, DomainGate
from rag_v2.agent.guardrail import EvidenceGuardrail, EvidenceGuardrailConfig
from rag_v2.agent.llm_composer import ComposeResult, DeepSeekAnswerComposer
from rag_v2.agent.models import AnswerTrace, EvidenceItem, RagAnswer
from rag_v2.agent.verifier import CitationVerifier
from rag_v2.retrieval.multi_query_pipeline import MultiQueryPipelineOptions, MultiQuerySearchOutput


class Stage5SearchPipeline(Protocol):
    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        options: MultiQueryPipelineOptions | None = None,
    ) -> MultiQuerySearchOutput: ...


class ContextPackingPipeline(Protocol):
    def pack(self, candidates: list[Any], *, token_budget: int = 1200, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class RagAnswerServiceOptions:
    """Runtime knobs for Stage6 answer service."""

    top_k: int = 5
    dense_top_k: int = 30
    sparse_top_k: int = 30
    branch_candidate_top_k: int = 10
    max_branches: int | None = None
    token_budget: int = 1200
    enable_reranker: bool | None = None
    enable_mmr: bool | None = None
    rerank_top_k: int = 30
    mmr_pre_candidates: int = 20
    mmr_top_k: int | None = None
    fallback_answer: str = "当前知识库没有足够依据回答该问题。"
    min_evidence_count: int = 1
    min_evidence_score: float = 0.3
    composer_mode: str = "template"

    def validate(self) -> None:
        for name in (
            "top_k",
            "dense_top_k",
            "sparse_top_k",
            "branch_candidate_top_k",
            "token_budget",
            "rerank_top_k",
            "mmr_pre_candidates",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.mmr_top_k is not None and self.mmr_top_k < 0:
            raise ValueError("mmr_top_k must be non-negative")
        if self.min_evidence_count < 0:
            raise ValueError("min_evidence_count must be non-negative")
        if self.min_evidence_score < 0:
            raise ValueError("min_evidence_score must be non-negative")
        if self.composer_mode not in {"template", "deepseek"}:
            raise ValueError("composer_mode must be template or deepseek")

    def to_stage5_options(self) -> MultiQueryPipelineOptions:
        self.validate()
        return MultiQueryPipelineOptions(
            top_k=self.top_k,
            dense_top_k=self.dense_top_k,
            sparse_top_k=self.sparse_top_k,
            rerank_top_k=self.rerank_top_k,
            mmr_pre_candidates=self.mmr_pre_candidates,
            mmr_top_k=self.mmr_top_k if self.mmr_top_k is not None else self.top_k,
            enable_reranker=self.enable_reranker,
            enable_mmr=self.enable_mmr,
            branch_candidate_top_k=self.branch_candidate_top_k,
            max_branches=self.max_branches,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RagAnswerService:
    """Reusable Stage6 service that wraps Stage5 retrieval output as RagAnswer."""

    def __init__(
        self,
        stage5_pipeline: Stage5SearchPipeline,
        *,
        context_packer: ContextPackingPipeline | None = None,
        default_options: RagAnswerServiceOptions | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        composer: TemplateAnswerComposer | None = None,
        deepseek_composer: DeepSeekAnswerComposer | None = None,
        verifier: CitationVerifier | None = None,
        guardrail: EvidenceGuardrail | None = None,
        domain_gate: DomainGate | None = None,
    ):
        self.stage5_pipeline = stage5_pipeline
        self.context_packer = context_packer
        self.default_options = default_options or RagAnswerServiceOptions()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.composer = composer or TemplateAnswerComposer(
            low_confidence_threshold=self.default_options.min_evidence_score,
            fallback_answer=self.default_options.fallback_answer,
        )
        self.deepseek_composer = deepseek_composer
        self.verifier = verifier or CitationVerifier()
        self.guardrail = guardrail or EvidenceGuardrail(
            EvidenceGuardrailConfig(
                min_evidence_count=self.default_options.min_evidence_count,
                min_score=self.default_options.min_evidence_score,
                fallback_answer=self.default_options.fallback_answer,
            )
        )
        self.domain_gate = domain_gate or DomainGate()

    @classmethod
    def from_config(
        cls,
        *,
        cfg: Any | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        reranker_device: str | None = None,
        fake_reranker: bool = False,
        hyde_mode: str = "disabled",
        enable_reranker: bool = True,
        enable_mmr: bool = True,
        default_options: RagAnswerServiceOptions | None = None,
    ) -> "RagAnswerService":
        """Build a real Stage5 stack from project artifacts.

        Imports are intentionally local to avoid loading optional dependencies
        when tests use a fake pipeline.
        """

        import json

        from rag_v2.config import get_config
        from rag_v2.query.analyzer import QueryAnalyzer
        from rag_v2.query.hyde import DeepSeekHydeGenerator, DisabledHydeGenerator, RuleBasedHydeGenerator
        from rag_v2.query.rewriter import QueryRewriter
        from rag_v2.retrieval.bm25_index import BM25Index
        from rag_v2.retrieval.context_packer import ContextPacker
        from rag_v2.retrieval.hybrid_searcher import HybridSearcher
        from rag_v2.retrieval.mmr import MMRSelector
        from rag_v2.retrieval.multi_query_pipeline import MultiQueryPipeline
        from rag_v2.retrieval.qdrant_searcher import QdrantSearcher
        from rag_v2.retrieval.rerank_pipeline import RerankPipeline
        from rag_v2.retrieval.reranker import CrossEncoderReranker, FakeReranker

        cfg = cfg or get_config()
        metadata_rows = json.loads((cfg.stage1_artifacts_dir / "chunk_metadata.json").read_text(encoding="utf-8"))
        dense = QdrantSearcher.from_config(cfg=cfg, device=device, batch_size=batch_size)
        sparse = BM25Index.load(cfg.artifacts_dir / "stage3" / "bm25_index.json")
        hybrid = HybridSearcher(dense_searcher=dense, sparse_index=sparse)

        reranker = None
        if enable_reranker:
            if fake_reranker:
                reranker = FakeReranker(score_fn=lambda _query, candidate: candidate.fusion_score)
            else:
                model_ref = cfg.reranker_model_path or cfg.reranker_model_name
                reranker = CrossEncoderReranker(
                    str(model_ref),
                    device=reranker_device or cfg.reranker_device,
                    batch_size=cfg.reranker_batch_size,
                    max_length=cfg.reranker_max_length,
                )
        mmr_selector = MMRSelector(dense.embedder) if enable_mmr else None
        stage4 = RerankPipeline.from_components(hybrid_searcher=hybrid, reranker=reranker, mmr_selector=mmr_selector)

        if hyde_mode == "rule":
            hyde_generator = RuleBasedHydeGenerator()
        elif hyde_mode == "deepseek":
            hyde_generator = DeepSeekHydeGenerator(
                api_key=cfg.deepseek_api_key,
                model=cfg.deepseek_model,
                timeout=cfg.deepseek_timeout,
                max_retries=cfg.deepseek_max_retries,
                fallback=RuleBasedHydeGenerator(),
            )
        else:
            hyde_generator = DisabledHydeGenerator()

        stage5 = MultiQueryPipeline(stage4, analyzer=QueryAnalyzer(), rewriter=QueryRewriter(), hyde_generator=hyde_generator)
        effective_options = default_options or RagAnswerServiceOptions()
        template_composer = TemplateAnswerComposer(
            low_confidence_threshold=effective_options.min_evidence_score,
            fallback_answer=effective_options.fallback_answer,
        )
        deepseek_composer = None
        domain_gate = DomainGate()
        if cfg.deepseek_api_key:
            from rag_v2.llm.deepseek_client import DeepSeekChatClient

            deepseek_client = DeepSeekChatClient(
                    api_key=cfg.deepseek_api_key,
                    model=cfg.deepseek_model,
                    timeout=max(cfg.deepseek_timeout, 20),
                    max_retries=cfg.deepseek_max_retries,
                )
            deepseek_composer = DeepSeekAnswerComposer(
                client=deepseek_client,
                fallback=template_composer,
            )
            domain_gate = DomainGate(classifier=DeepSeekDomainClassifier(deepseek_client))
        return cls(
            stage5,
            context_packer=ContextPacker(metadata_rows),
            default_options=effective_options,
            composer=template_composer,
            deepseek_composer=deepseek_composer,
            domain_gate=domain_gate,
        )

    def answer(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        options: RagAnswerServiceOptions | None = None,
    ) -> RagAnswer:
        options = options or self.default_options
        options.validate()
        start = time.perf_counter()
        try:
            output = self.stage5_pipeline.search(query, filters=filters, options=options.to_stage5_options())
            evidence = self._build_evidence(output, token_budget=options.token_budget)
            citations = [item.to_citation() for item in evidence]
            elapsed_ms = (time.perf_counter() - start) * 1000
            domain_result = self.domain_gate.evaluate(query, evidence)
            if domain_result.decision == "out_of_domain":
                trace = self._build_trace(
                    output,
                    elapsed_ms=elapsed_ms,
                    options=options,
                    evidence=evidence,
                    compose_result=ComposeResult(
                        options.fallback_answer,
                        options.composer_mode,
                        "domain_gate",
                        0.0,
                        "out_of_domain",
                        "fallback",
                        (),
                        "domain_gate",
                    ),
                    domain_gate=domain_result.to_dict(),
                    verification={"passed": True, "reasons": []},
                    guardrail={"passed": False, "reasons": ["out_of_domain"]},
                )
                return RagAnswer(
                    query=query,
                    answer=options.fallback_answer,
                    citations=[],
                    evidence=evidence,
                    trace=trace,
                    fallback_reason="out_of_domain",
                )
            compose_result = self._compose(query, evidence, options)
            if compose_result.decision == "fallback":
                trace = self._build_trace(
                    output,
                    elapsed_ms=elapsed_ms,
                    options=options,
                    evidence=evidence,
                    compose_result=compose_result,
                    domain_gate=domain_result.to_dict(),
                    verification={"passed": True, "reasons": []},
                    guardrail={"passed": False, "reasons": [compose_result.fallback_reason or "insufficient_evidence"]},
                )
                return RagAnswer(
                    query=query,
                    answer=compose_result.answer,
                    citations=[],
                    evidence=evidence,
                    trace=trace,
                    fallback_reason=compose_result.fallback_reason or "insufficient_evidence",
                )
            draft_answer, verification = self.verifier.verify_with_repair(
                compose_result.answer, citations, evidence
            )
            guardrail = self.guardrail.check(evidence, verification)
            trace = self._build_trace(
                output,
                elapsed_ms=elapsed_ms,
                options=options,
                evidence=evidence,
                compose_result=compose_result,
                verification=verification.to_dict(),
                guardrail=guardrail.to_dict(),
                domain_gate=domain_result.to_dict(),
            )
            if not guardrail.passed:
                return RagAnswer(
                    query=query,
                    answer=guardrail.fallback_answer or options.fallback_answer,
                    citations=citations,
                    evidence=evidence,
                    trace=trace,
                    fallback_reason=guardrail.reasons[0] if guardrail.reasons else "guardrail_failed",
                )
            return RagAnswer(
                query=query,
                answer=draft_answer,
                citations=citations,
                evidence=evidence,
                trace=trace,
                fallback_reason=None,
            )
        except Exception as exc:  # pragma: no cover - exercised by unit tests through behavior, not branch internals
            elapsed_ms = (time.perf_counter() - start) * 1000
            return RagAnswer(
                query=query,
                answer=options.fallback_answer,
                citations=[],
                evidence=[],
                trace=AnswerTrace(
                    retrieval_latency_ms=elapsed_ms,
                    composer_mode="fallback",
                    verification={"passed": False, "reasons": ["retrieval_exception"]},
                    extra={"exception_type": type(exc).__name__, "exception_message": str(exc)},
                ),
                fallback_reason="retrieval_exception",
            )

    def _build_evidence(self, output: MultiQuerySearchOutput, *, token_budget: int) -> list[EvidenceItem]:
        if self.context_packer is not None:
            packed = self.context_packer.pack(output.results, token_budget=token_budget)
            return self.evidence_builder.build_from_context_chunks(packed.evidence_chunks)
        return self.evidence_builder.build_from_search_results(output.results)

    def _compose(
        self,
        query: str,
        evidence: list[EvidenceItem],
        options: RagAnswerServiceOptions,
    ) -> ComposeResult:
        if not evidence:
            return ComposeResult(
                options.fallback_answer,
                options.composer_mode,
                "template",
                0.0,
                "insufficient_evidence",
                "fallback",
                (),
                "empty_evidence",
            )
        if options.composer_mode == "deepseek":
            if self.deepseek_composer is not None:
                return self.deepseek_composer.compose_with_trace(query, evidence)
            return ComposeResult(
                self.composer.compose(query, evidence),
                "deepseek",
                "template",
                0.0,
                "deepseek_composer_unavailable",
                "answered",
                tuple(item.citation_id for item in evidence),
                "template_fallback",
            )
        return ComposeResult(
            self.composer.compose(query, evidence),
            "template",
            "template",
            0.0,
            None,
            "answered",
            tuple(item.citation_id for item in evidence),
            "template",
        )

    def _build_trace(
        self,
        output: MultiQuerySearchOutput,
        *,
        elapsed_ms: float,
        options: RagAnswerServiceOptions,
        evidence: list[EvidenceItem],
        compose_result: ComposeResult,
        verification: dict[str, Any] | None = None,
        guardrail: dict[str, Any] | None = None,
        domain_gate: dict[str, Any] | None = None,
    ) -> AnswerTrace:
        matched = []
        for result in output.results:
            for branch in result.metadata.get("matched_branches", []) if result.metadata else []:
                if branch not in matched:
                    matched.append(branch)
        return AnswerTrace(
            query_plan=output.query_plan.to_dict(),
            branches=[branch.to_dict() for branch in output.query_branches],
            matched_branches=matched,
            retrieval_latency_ms=elapsed_ms,
            composer_mode=compose_result.actual_mode,
            verification={"passed": bool((guardrail or {}).get("passed", False)), "reasons": list((guardrail or {}).get("reasons", [])), "citation_verification": verification or {}, "guardrail": guardrail or {}},
            extra={
                "service_options": options.to_dict(),
                "result_count": len(output.results),
                "raw_retrieval_results": [
                    {
                        "rank": item.rank,
                        "source_file": item.source_file,
                        "chunk_id": item.chunk_id,
                        "score": (
                            item.rerank_score
                            if item.rerank_score is not None
                            else item.dense_score
                            if item.dense_score is not None
                            else item.fusion_score
                        ),
                        "rerank_score": item.rerank_score,
                        "dense_score": item.dense_score,
                        "fusion_score": item.fusion_score,
                        "mmr_score": item.mmr_score,
                    }
                    for item in output.results
                ],
                "evidence_count": len(evidence),
                "composer": compose_result.to_dict(),
                "hyde_document": output.hyde_document.to_dict() if output.hyde_document else None,
                "domain_gate": domain_gate or {},
                "answer_decision": {
                    "decision": compose_result.decision,
                    "decision_source": compose_result.decision_source,
                    "fallback_reason": compose_result.fallback_reason,
                    "used_citation_ids": list(compose_result.used_citation_ids),
                },
            },
        )
