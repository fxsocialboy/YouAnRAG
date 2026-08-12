from rag_v2.query.analyzer import QueryAnalyzer, normalize_query_text
from rag_v2.query.models import RetrievalPolicy


def test_query_analyzer_exact_fact_response_level_no_hyde():
    plan = QueryAnalyzer().analyze("IV级气象灾害应急响应一般由谁启动？")

    assert plan.query_type == "exact_fact"
    assert plan.hazard_hint == "气象灾害"
    assert any(term.upper().startswith("IV") for term in plan.extracted_terms)
    assert plan.retrieval_policy.use_hyde is False
    assert plan.retrieval_policy.use_multi_query is False
    assert plan.retrieval_policy.use_reranker is False
    assert plan.retrieval_policy.sparse_top_k >= plan.retrieval_policy.dense_top_k


def test_query_analyzer_scenario_enables_rewrite_and_multi_query():
    plan = QueryAnalyzer().analyze("台风黄色预警下学校应该怎么做")

    assert plan.query_type == "scenario"
    assert plan.hazard_hint == "台风"
    assert "学校" in plan.extracted_terms
    assert plan.retrieval_policy.use_rewrite is True
    assert plan.retrieval_policy.use_multi_query is True
    assert plan.retrieval_policy.use_mmr is True
    assert plan.retrieval_policy.max_branches == 3


def test_query_analyzer_year_and_magnitude_query_preserves_exactness():
    plan = QueryAnalyzer().analyze("2016年大陆5级以上地震损失情况在哪里能看到？")

    assert plan.query_type == "exact_fact"
    assert plan.hazard_hint == "地震"
    assert plan.time_hint.startswith("2016")
    assert "2016年" in plan.extracted_terms or "2016" in plan.extracted_terms
    assert plan.retrieval_policy.use_hyde is False
    assert plan.retrieval_policy.use_rewrite is False


def test_query_analyzer_short_query_is_ambiguous_but_safe():
    plan = QueryAnalyzer().analyze("洪水怎么办")

    assert plan.query_type in {"scenario", "ambiguous"}
    assert plan.hazard_hint == "洪涝"
    assert plan.retrieval_policy.use_hyde is False
    assert plan.retrieval_policy.branch_weights["raw"] == 1.0
    assert plan.retrieval_policy.branch_weights["hyde"] == 0.6


def test_query_analyzer_empty_query_falls_back_without_models():
    plan = QueryAnalyzer().analyze("  ")

    assert plan.query_type == "ambiguous"
    assert plan.normalized_query == ""
    assert plan.rewrite_confidence == 0.0
    assert plan.retrieval_policy.use_reranker is False
    assert "empty_query" in plan.reasons


def test_query_plan_to_dict_is_json_ready():
    plan = QueryAnalyzer().analyze("深圳地震后高层居民如何疏散")
    data = plan.to_dict()

    assert data["original_query"] == "深圳地震后高层居民如何疏散"
    assert data["retrieval_policy"]["branch_weights"]["normalized"] == 0.9
    assert data["region_hint"] == "深圳"


def test_normalize_query_text_unifies_fullwidth_and_roman_level():
    assert normalize_query_text("  Ⅳ 级 响应  ") == "IV 级 响应"
    assert normalize_query_text("ＩＶ级") == "IV级"


def test_retrieval_policy_default_branch_weights():
    policy = RetrievalPolicy()

    assert policy.branch_weights == {
        "raw": 1.0,
        "normalized": 0.9,
        "expanded": 0.7,
        "hyde": 0.6,
    }
