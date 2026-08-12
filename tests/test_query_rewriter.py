from rag_v2.query.analyzer import QueryAnalyzer
from rag_v2.query.rewriter import QueryRewriter, build_expanded_query


def branch_map(branches):
    return {item.branch: item for item in branches}


def test_rewriter_always_keeps_raw_query_first():
    plan = QueryAnalyzer().analyze("台风黄色预警下学校应该怎么做")
    branches = QueryRewriter().rewrite(plan)

    assert branches[0].branch == "raw"
    assert branches[0].query == "台风黄色预警下学校应该怎么做"
    assert branches[0].weight == 1.0


def test_rewriter_generates_expanded_branch_for_scenario_query():
    plan = QueryAnalyzer().analyze("台风黄色预警下学校应该怎么做")
    branches = branch_map(QueryRewriter().rewrite(plan))

    assert "expanded" in branches
    expanded = branches["expanded"].query
    assert "台风" in expanded
    assert "热带气旋" in expanded
    assert "学校" in expanded
    assert "防御措施" in expanded
    assert branches["expanded"].weight == 0.7


def test_rewriter_does_not_over_expand_exact_fact_query():
    plan = QueryAnalyzer().analyze("IV级气象灾害应急响应一般由谁启动？")
    branches = QueryRewriter().rewrite(plan)

    assert [item.branch for item in branches] == ["raw"]
    assert branches[0].query == "IV级气象灾害应急响应一般由谁启动？"


def test_rewriter_adds_normalized_when_text_changes():
    plan = QueryAnalyzer().analyze("  Ⅳ 级 响应  ")
    branches = branch_map(QueryRewriter().rewrite(plan))

    assert "raw" in branches
    assert "normalized" in branches
    assert branches["normalized"].query == "IV 级 响应"
    assert branches["normalized"].weight == 0.9


def test_rewriter_deduplicates_raw_and_normalized():
    plan = QueryAnalyzer().analyze("地震后高层居民如何疏散")
    branches = QueryRewriter().rewrite(plan)
    queries = [item.query for item in branches]

    assert len(queries) == len(set(queries))
    assert "normalized" not in [item.branch for item in branches]


def test_rewriter_respects_max_branches():
    plan = QueryAnalyzer().analyze("地震后高层居民如何疏散")
    branches = QueryRewriter(max_branches=1).rewrite(plan)

    assert len(branches) == 1
    assert branches[0].branch == "raw"


def test_expanded_query_for_short_ambiguous_query_adds_safe_terms():
    plan = QueryAnalyzer().analyze("洪水怎么办")
    expanded = build_expanded_query(plan)

    assert "洪涝" in expanded
    assert "应急处置" in expanded
    assert "防灾减灾" in expanded or "转移避险" in expanded


def test_rewriter_empty_query_returns_no_empty_branch():
    plan = QueryAnalyzer().analyze("  ")
    branches = QueryRewriter().rewrite(plan)

    assert branches == []
