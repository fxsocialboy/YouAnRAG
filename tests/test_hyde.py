import pytest

from rag_v2.query.analyzer import QueryAnalyzer
from rag_v2.query.hyde import (
    DeepSeekHydeGenerator,
    DisabledHydeGenerator,
    FakeHydeGenerator,
    RuleBasedHydeGenerator,
    sanitize_hyde_content,
    should_use_hyde,
)


def test_disabled_hyde_generator_never_generates_content():
    plan = QueryAnalyzer().analyze("台风黄色预警下学校应该怎么做")
    doc = DisabledHydeGenerator().generate(plan.original_query, plan)

    assert doc.used is False
    assert doc.content == ""
    assert doc.reason == "hyde_disabled"
    assert doc.mode == "disabled"


def test_rule_based_hyde_generates_for_scenario_query():
    plan = QueryAnalyzer().analyze("台风黄色预警下学校应该怎么做")
    doc = RuleBasedHydeGenerator().generate(plan.original_query, plan)

    assert doc.used is True
    assert doc.mode == "rule"
    assert "台风" in doc.content
    assert "应急" in doc.content
    assert len(doc.content) <= 220


def test_rule_based_hyde_skips_exact_fact_query():
    plan = QueryAnalyzer().analyze("IV级气象灾害应急响应一般由谁启动？")
    doc = RuleBasedHydeGenerator().generate(plan.original_query, plan)

    assert should_use_hyde(plan) is False
    assert doc.used is False
    assert doc.content == ""
    assert "not_allowed" in doc.reason


def test_fake_hyde_is_deterministic_for_tests():
    plan = QueryAnalyzer().analyze("洪水怎么办")
    doc = FakeHydeGenerator(content="固定 HyDE 文本").generate(plan.original_query, plan)

    assert doc.used is True
    assert doc.content == "固定 HyDE 文本"
    assert doc.mode == "fake"


def test_deepseek_hyde_requires_api_key():
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekHydeGenerator(api_key=None)


def test_deepseek_hyde_falls_back_when_request_fails(monkeypatch):
    plan = QueryAnalyzer().analyze("台风黄色预警下学校应该怎么做")
    generator = DeepSeekHydeGenerator(
        api_key="test-key",
        timeout=1,
        max_retries=0,
        fallback=RuleBasedHydeGenerator(),
    )
    def fail_call(self, query):
        raise RuntimeError("boom")

    monkeypatch.setattr(DeepSeekHydeGenerator, "_call_deepseek", fail_call)
    doc = generator.generate(plan.original_query, plan)

    assert doc.used is True
    assert doc.mode == "deepseek->rule"
    assert "deepseek_failed_fallback" in doc.reason
    assert "台风" in doc.content


def test_sanitize_hyde_content_compacts_and_limits():
    text = sanitize_hyde_content("  第一行\n\n第二行  " + "x" * 300, max_chars=20)

    assert "\n" not in text
    assert len(text) == 20
