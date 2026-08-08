from rag_v2.ingestion.normalizer import (
    has_english_word_gluing,
    normalize_lines,
    normalize_markdown,
)


def test_preserves_english_word_spaces():
    raw = "Climate   change risk and disaster   response"
    normalized = normalize_markdown(raw)
    assert normalized == "Climate change risk and disaster response"
    assert "Climatechange" not in normalized
    assert not has_english_word_gluing(raw, normalized)


def test_preserves_markdown_heading_and_newline_structure():
    raw = "# 预案总则\r\n\r\n## 4 应急响应\r\n气象部门加强监测预报。"
    normalized = normalize_markdown(raw)
    assert "# 预案总则" in normalized
    assert "## 4 应急响应" in normalized
    assert "\n## 4 应急响应\n" in normalized
    assert "预案总则##" not in normalized


def test_preserves_list_and_clause_markers():
    raw = "（一）启动条件\n1. 预警发布\n① 转移群众\nⅣ级响应"
    normalized = normalize_markdown(raw)
    assert "（一）启动条件" in normalized
    assert "1. 预警发布" in normalized
    assert "① 转移群众" in normalized
    assert "Ⅳ级响应" in normalized


def test_decodes_html_entities():
    raw = "风险 &amp; 应急 &lt;预案&gt;"
    assert normalize_markdown(raw) == "风险 & 应急 <预案>"


def test_removes_control_characters_but_keeps_newlines():
    raw = "第一行\x00\x08\n第二行\x7f"
    normalized = normalize_markdown(raw)
    assert normalized == "第一行\n第二行"


def test_compresses_blank_lines_and_trailing_spaces():
    raw = "标题   \n\n\n\n正文   \n"
    normalized = normalize_markdown(raw)
    assert normalized == "标题\n\n正文"
    assert "\n\n\n" not in normalized


def test_normalize_lines_returns_non_empty_stripped_lines():
    raw = "  # 标题  \n\n  - 条目一  \n  正文  "
    assert normalize_lines(raw) == ["# 标题", "- 条目一", "正文"]


def test_none_and_non_string_inputs_are_handled():
    assert normalize_markdown(None) == ""
    assert normalize_markdown(123) == "123"


def test_has_english_word_gluing_detects_legacy_style_bug():
    before = "Climate change adaptation"
    after = "Climatechangeadaptation"
    assert has_english_word_gluing(before, after)
