import pytest

from rag_v2.agent.service import RagAnswerService, RagAnswerServiceOptions
from rag_v2.agent.composer import TemplateAnswerComposer
from rag_v2.agent.llm_composer import DeepSeekAnswerComposer
from rag_v2.agent.domain_gate import DomainGate
from rag_v2.query.models import QueryBranch, QueryPlan
from rag_v2.retrieval.hybrid_searcher import HybridSearchResult
from rag_v2.retrieval.multi_query_pipeline import MultiQuerySearchOutput


class FakeStage5Pipeline:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search(self, query, *, filters=None, options=None):
        self.calls.append({"query": query, "filters": filters, "options": options})
        if self.error:
            raise self.error
        return MultiQuerySearchOutput(
            query_plan=QueryPlan(original_query=query, normalized_query=query, query_type="scenario"),
            query_branches=[QueryBranch(branch="raw", query=query, weight=1.0, reason="raw_query")],
            results=self.results,
            hyde_document=None,
        )


def make_result(rank=1, chunk_id="a.md::0", source_file="a.md", content="学校应停止户外活动。"):
    return HybridSearchResult(
        chunk_id=chunk_id,
        content=content,
        source_file=source_file,
        chunk_index=0,
        section_path_text="一、响应措施",
        dense_score=0.8,
        sparse_score=0.7,
        fusion_score=0.9,
        rank=rank,
        metadata={"matched_branches": ["raw"], "token_count": 20},
        rerank_score=0.95,
        mmr_score=0.88,
        stage4_rank=rank,
    )


def test_rag_answer_service_wraps_fake_stage5_results():
    pipeline = FakeStage5Pipeline(results=[make_result(), make_result(rank=2, chunk_id="b.md::1", source_file="b.md")])
    service = RagAnswerService(pipeline)

    answer = service.answer("台风黄色预警下学校应该怎么做", options=RagAnswerServiceOptions(top_k=2))

    assert answer.is_fallback is False
    assert answer.query == "台风黄色预警下学校应该怎么做"
    assert answer.evidence[0].citation_id == "S1"
    assert answer.citations[1].marker == "[S2]"
    assert "[S1]" in answer.answer and "[S2]" in answer.answer
    assert answer.trace.query_plan["query_type"] == "scenario"
    assert answer.trace.branches[0]["branch"] == "raw"
    assert answer.trace.matched_branches == ["raw"]
    assert pipeline.calls[0]["options"].top_k == 2


def test_rag_answer_service_returns_fallback_on_empty_results():
    service = RagAnswerService(FakeStage5Pipeline(results=[]))

    answer = service.answer("知识库没有的问题", options=RagAnswerServiceOptions(top_k=3))

    assert answer.is_fallback is True
    assert answer.fallback_reason == "insufficient_evidence"
    assert answer.citations == []
    assert answer.trace.verification["passed"] is False
    assert "insufficient_evidence" in answer.trace.verification["reasons"]


def test_rag_answer_service_catches_pipeline_exception():
    service = RagAnswerService(FakeStage5Pipeline(error=RuntimeError("boom")))

    answer = service.answer("台风怎么办")

    assert answer.is_fallback is True
    assert answer.fallback_reason == "retrieval_exception"
    assert answer.trace.composer_mode == "fallback"
    assert answer.trace.extra["exception_type"] == "RuntimeError"
    assert "boom" in answer.trace.extra["exception_message"]


def test_rag_answer_service_options_validate_and_convert_to_stage5_options():
    options = RagAnswerServiceOptions(top_k=4, mmr_top_k=None, enable_reranker=False, enable_mmr=True)
    stage5_options = options.to_stage5_options()

    assert stage5_options.top_k == 4
    assert stage5_options.mmr_top_k == 4
    assert stage5_options.enable_reranker is False
    assert stage5_options.enable_mmr is True


class FakeChatClient:
    def __init__(self, response):
        self.response = response

    def complete(self, messages, **kwargs):
        return self.response


def test_rag_answer_service_selects_deepseek_composer_and_records_trace():
    template = TemplateAnswerComposer()
    deepseek = DeepSeekAnswerComposer(
        client=FakeChatClient("学校应停止户外活动。[S1]"),
        fallback=template,
    )
    service = RagAnswerService(
        FakeStage5Pipeline(results=[make_result()]),
        composer=template,
        deepseek_composer=deepseek,
    )

    answer = service.answer(
        "台风黄色预警下学校应该怎么做",
        options=RagAnswerServiceOptions(composer_mode="deepseek"),
    )

    assert answer.is_fallback is False
    assert answer.trace.composer_mode == "deepseek"
    assert answer.trace.extra["composer"]["requested_mode"] == "deepseek"
    assert answer.trace.extra["composer"]["actual_mode"] == "deepseek"


def test_rag_answer_service_falls_back_when_deepseek_composer_is_unavailable():
    service = RagAnswerService(FakeStage5Pipeline(results=[make_result()]))

    answer = service.answer("学校怎么办", options=RagAnswerServiceOptions(composer_mode="deepseek"))

    assert answer.is_fallback is False
    assert answer.trace.composer_mode == "template"
    assert answer.trace.extra["composer"]["fallback_reason"] == "deepseek_composer_unavailable"


@pytest.mark.parametrize("field", ["top_k", "dense_top_k", "token_budget"])
def test_rag_answer_service_options_reject_invalid_values(field):
    kwargs = {field: -1}
    if field == "token_budget":
        kwargs[field] = 0
    options = RagAnswerServiceOptions(**kwargs)

    with pytest.raises(ValueError):
        options.validate()


def test_rag_answer_service_options_reject_invalid_composer_mode():
    with pytest.raises(ValueError, match="composer_mode"):
        RagAnswerServiceOptions(composer_mode="unknown").validate()


def test_rag_answer_service_short_circuits_ood_before_composer():
    client = FakeChatClient('{"decision":"answered","answer":"不应被调用。[S1]","fallback_reason":null,"used_citation_ids":["S1"]}')
    template = TemplateAnswerComposer()
    service = RagAnswerService(
        FakeStage5Pipeline(results=[make_result()]),
        composer=template,
        deepseek_composer=DeepSeekAnswerComposer(client=client, fallback=template),
        domain_gate=DomainGate(),
    )
    answer = service.answer("怎样优化CUDA矩阵乘法内核？", options=RagAnswerServiceOptions(composer_mode="deepseek"))
    assert answer.is_fallback is True
    assert answer.decision == "fallback"
    assert answer.fallback_reason == "out_of_domain"
    assert answer.citations == []
    assert answer.trace.extra["domain_gate"]["decision"] == "out_of_domain"
    assert answer.trace.extra["answer_decision"]["decision_source"] == "domain_gate"


def test_rag_answer_service_keeps_structured_composer_fallback_consistent():
    template = TemplateAnswerComposer()
    deepseek = DeepSeekAnswerComposer(
        client=FakeChatClient('{"decision":"fallback","answer":"当前证据无法支持结论。","fallback_reason":"insufficient_evidence","used_citation_ids":[]}'),
        fallback=template,
    )
    service = RagAnswerService(FakeStage5Pipeline(results=[make_result()]), composer=template, deepseek_composer=deepseek)
    answer = service.answer("台风下学校怎么办", options=RagAnswerServiceOptions(composer_mode="deepseek"))
    assert answer.decision == "fallback"
    assert answer.is_fallback is True
    assert answer.fallback_reason == "insufficient_evidence"
    assert answer.citations == []
    assert answer.trace.extra["answer_decision"]["decision"] == "fallback"
