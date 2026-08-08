from pathlib import Path

from rag_v2.ingestion.markdown_parser import (
    MarkdownBlock,
    classify_line,
    parse_markdown,
    parse_markdown_file,
    section_path_coverage,
)


def indexable_blocks(blocks):
    return [b for b in blocks if b.is_indexable]


def test_heading_stack_inherits_parent_section_path():
    md = """# 深圳市气象灾害应急预案

## 4 应急响应

### 4.4 分灾种响应

气象部门加强监测预报。
"""
    blocks = parse_markdown(md, "深圳市气象灾害应急预案.md")
    body = indexable_blocks(blocks)[0]
    assert body.block_type == "paragraph"
    assert body.section_path == ["深圳市气象灾害应急预案", "4 应急响应", "4.4 分灾种响应"]
    assert body.text == "气象部门加强监测预报。"


def test_same_level_heading_replaces_previous_peer():
    md = """# 文档
## 第一章
第一章正文
## 第二章
第二章正文
"""
    bodies = indexable_blocks(parse_markdown(md, "文档.md"))
    assert bodies[0].section_path == ["文档", "第一章"]
    assert bodies[1].section_path == ["文档", "第二章"]


def test_lower_heading_inherits_current_parent():
    md = """# 文档
## 4 应急响应
### 4.1 启动条件
达到条件后启动。
"""
    body = indexable_blocks(parse_markdown(md, "文档.md"))[0]
    assert body.section_path == ["文档", "4 应急响应", "4.1 启动条件"]


def test_file_name_used_as_default_root_when_no_h1():
    md = """## 总则
本预案适用于自然灾害救助。
"""
    body = indexable_blocks(parse_markdown(md, "吉林省自然灾害救助应急预案.md"))[0]
    assert body.section_path == ["吉林省自然灾害救助应急预案", "总则"]


def test_list_lines_are_grouped_and_indexable():
    md = """# 文档
## 任务
（一）启动条件
1. 发布预警
① 转移群众
"""
    blocks = indexable_blocks(parse_markdown(md, "文档.md"))
    assert len(blocks) == 1
    assert blocks[0].block_type == "list"
    assert "① 转移群众" in blocks[0].text
    assert blocks[0].section_path == ["文档", "任务"]


def test_toc_lines_are_marked_non_indexable():
    md = """# 文档
1.1 总体目标 15
1.2 主要任务 16
## 正文
这里是真正正文。
"""
    blocks = parse_markdown(md, "文档.md")
    toc_blocks = [b for b in blocks if b.block_type == "toc"]
    assert toc_blocks
    assert all(not b.is_indexable for b in toc_blocks)
    assert indexable_blocks(blocks)[0].text == "这里是真正正文。"


def test_reference_section_marked_non_indexable():
    md = """# 论文
## 参考文献
[1] Zhang A. Journal of Disaster, doi:10.1/abc
[2] 王某某. 灾害研究学报.
## 结论
遥感可以用于灾损评估。
"""
    blocks = parse_markdown(md, "论文.md")
    refs = [b for b in blocks if b.block_type == "reference"]
    assert refs
    assert all(not b.is_indexable for b in refs)
    assert indexable_blocks(blocks)[0].text == "遥感可以用于灾损评估。"


def test_table_lines_are_grouped_as_table():
    md = """# 文档
| 指标 | 单位 |
| 因灾死亡人口 | 人 |
"""
    block = indexable_blocks(parse_markdown(md, "文档.md"))[0]
    assert block.block_type == "table"
    assert "因灾死亡人口" in block.text


def test_parse_markdown_file_reads_utf8_file():
    tmp_dir = Path(r"G:\tiaozhanbei\newrag\artifacts\stage1\test_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / "测试预案.md"
    try:
        p.write_text("# 测试预案\n正文。", encoding="utf-8")
        blocks = parse_markdown_file(p)
        assert indexable_blocks(blocks)[0].source_file == "测试预案.md"
        assert indexable_blocks(blocks)[0].section_path == ["测试预案"]
    finally:
        if p.exists():
            p.unlink()


def test_block_to_dict_and_coverage():
    blocks = parse_markdown("# 文档\n## 章节\n正文。", "文档.md")
    body = indexable_blocks(blocks)[0]
    d = body.to_dict()
    assert d["source_file"] == "文档.md"
    assert d["section_path"] == ["文档", "章节"]
    assert section_path_coverage(blocks) == 1.0


def test_classify_line_basic_cases():
    assert classify_line("| 字段 | 值 |") == "table"
    assert classify_line("1.1 总体目标 15") == "toc"
    assert classify_line("（一）启动条件") == "list"
    assert classify_line("[1] Zhang A. Journal of Disaster") == "reference"
    assert classify_line("气象部门加强监测预报。") == "paragraph"
