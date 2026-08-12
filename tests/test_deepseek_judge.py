import json

import pytest

from rag_v2.agent.models import EvidenceItem
from rag_v2.evaluation.deepseek_judge import DeepSeekAnswerJudge, parse_json_object


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.response


def make_evidence():
    return [EvidenceItem("S1", "a.md::1", "a.md", ["响应"], "Ⅳ级响应由副局长签发。", 0.9, 1)]


def test_deepseek_judge_parses_atomic_fact_scores():
    response = json.dumps(
        {
            "atomic_facts": [
                {
                    "fact": "Ⅳ级响应由副局长签发",
                    "supported": True,
                    "cited": True,
                    "supporting_citation_ids": ["S1"],
                    "reason": "S1明确支持",
                },
                {
                    "fact": "必须在十分钟内启动",
                    "supported": False,
                    "cited": False,
                    "supporting_citation_ids": [],
                    "reason": "证据未给出时限",
                },
            ],
            "answer_relevancy": 0.9,
            "reason": "回答了签发主体",
        },
        ensure_ascii=False,
    )
    client = FakeClient(response)
    result = DeepSeekAnswerJudge(client).evaluate(
        query="Ⅳ级响应由谁签发？",
        answer="Ⅳ级响应由副局长签发[S1]，且必须十分钟内启动。",
        evidence=make_evidence(),
        composer_mode="deepseek",
    )

    assert result.faithfulness == 0.5
    assert result.citation_completeness == 0.5
    assert result.citation_correctness == 1.0
    assert result.answer_relevancy == 0.9
    assert client.calls[0]["kwargs"]["temperature"] == 0.0
    assert client.calls[0]["kwargs"]["response_format"] == {"type": "json_object"}


def test_parse_json_object_accepts_markdown_fence():
    assert parse_json_object('```json\n{"atomic_facts": []}\n```') == {"atomic_facts": []}


def test_deepseek_judge_rejects_missing_atomic_facts():
    judge = DeepSeekAnswerJudge(FakeClient('{"atomic_facts": [], "answer_relevancy": 0.5}'))
    with pytest.raises(ValueError, match="no atomic_facts"):
        judge.evaluate(query="q", answer="a", evidence=make_evidence(), composer_mode="deepseek")


def test_deepseek_judge_does_not_accept_unknown_supporting_citation():
    client = FakeClient(
        json.dumps(
            {
                "atomic_facts": [
                    {
                        "fact": "Ⅳ级响应由副局长签发",
                        "supported": True,
                        "cited": True,
                        "supporting_citation_ids": ["S999"],
                        "reason": "错误引用",
                    }
                ],
                "answer_relevancy": 1.0,
            },
            ensure_ascii=False,
        )
    )
    result = DeepSeekAnswerJudge(client).evaluate(
        query="Ⅳ级响应由谁签发？",
        answer="Ⅳ级响应由副局长签发[S999]",
        evidence=make_evidence(),
        composer_mode="deepseek",
    )

    assert result.faithfulness == 0.0
    assert "S999" in result.atomic_facts[0].reason
