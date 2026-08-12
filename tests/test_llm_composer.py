from rag_v2.agent.composer import TemplateAnswerComposer
from rag_v2.agent.llm_composer import DeepSeekAnswerComposer
from rag_v2.agent.models import EvidenceItem


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.error:
            raise self.error
        return self.response


def evidence():
    return [
        EvidenceItem(
            citation_id="S1",
            chunk_id="policy.md::1",
            source_file="policy.md",
            section_path=["Ⅳ级响应"],
            content="Ⅳ级应急响应原则上由副局长签发。",
            score=0.9,
            rank=1,
        )
    ]


def test_deepseek_composer_returns_llm_answer_and_trace():
    client = FakeClient("Ⅳ级应急响应原则上由副局长签发。[S1]")
    composer = DeepSeekAnswerComposer(client=client, fallback=TemplateAnswerComposer())

    result = composer.compose_with_trace("Ⅳ级响应由谁签发？", evidence())

    assert result.actual_mode == "deepseek"
    assert result.fallback_reason is None
    assert "[S1]" in result.answer
    prompt = client.calls[0]["messages"][1]["content"]
    assert "policy.md" in prompt and "Ⅳ级应急响应" in prompt


def test_deepseek_composer_falls_back_on_api_error():
    composer = DeepSeekAnswerComposer(
        client=FakeClient(error=TimeoutError("timeout")),
        fallback=TemplateAnswerComposer(),
    )

    result = composer.compose_with_trace("Ⅳ级响应由谁签发？", evidence())

    assert result.actual_mode == "template"
    assert result.fallback_reason == "deepseek_failed:TimeoutError"
    assert "[S1]" in result.answer


def test_deepseek_composer_does_not_call_api_without_evidence():
    client = FakeClient("should not be used")
    composer = DeepSeekAnswerComposer(client=client, fallback=TemplateAnswerComposer())

    result = composer.compose_with_trace("无证据问题", [])

    assert result.actual_mode == "template"
    assert result.fallback_reason == "empty_evidence"
    assert client.calls == []


def test_deepseek_composer_parses_structured_answered_decision():
    client = FakeClient('{"decision":"answered","answer":"由副局长签发。[S1]","fallback_reason":null,"used_citation_ids":["S1"]}')
    result = DeepSeekAnswerComposer(client=client, fallback=TemplateAnswerComposer()).compose_with_trace("谁签发？", evidence())
    assert result.decision == "answered"
    assert result.used_citation_ids == ("S1",)
    assert result.decision_source == "structured_llm"


def test_deepseek_composer_parses_structured_fallback_decision():
    client = FakeClient('{"decision":"fallback","answer":"现有证据不足。","fallback_reason":"insufficient_evidence","used_citation_ids":[]}')
    result = DeepSeekAnswerComposer(client=client, fallback=TemplateAnswerComposer()).compose_with_trace("未知问题", evidence())
    assert result.decision == "fallback"
    assert result.fallback_reason == "insufficient_evidence"


def test_structured_fallback_removes_citation_markers():
    client = FakeClient('{"decision":"fallback","answer":"现有证据不足。[S1]","fallback_reason":"insufficient_evidence","used_citation_ids":["S1"]}')
    result = DeepSeekAnswerComposer(client=client, fallback=TemplateAnswerComposer()).compose_with_trace("未知问题", evidence())
    assert result.decision == "fallback"
    assert result.used_citation_ids == ()
    assert "[S1]" not in result.answer


def test_deepseek_composer_invalid_json_degrades_to_template():
    result = DeepSeekAnswerComposer(
        client=FakeClient("不是合法JSON也没有引用"), fallback=TemplateAnswerComposer()
    ).compose_with_trace("谁签发？", evidence())
    assert result.actual_mode == "template"
    assert result.decision_source == "template_fallback"
    assert result.fallback_reason == "deepseek_failed:ValueError"


def test_partial_insufficiency_with_citation_is_not_global_fallback():
    result = DeepSeekAnswerComposer(
        client=FakeClient("部分细节现有证据不足，但Ⅳ级响应由副局长签发。[S1]"), fallback=TemplateAnswerComposer()
    ).compose_with_trace("谁签发？", evidence())
    assert result.decision == "answered"
    assert result.fallback_reason is None
