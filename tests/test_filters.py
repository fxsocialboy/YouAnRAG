from __future__ import annotations

from rag_v2.retrieval.filters import (
    build_source_file_catalog,
    detect_metadata_filters,
    detect_source_file_filter,
    normalize_text_for_match,
)


def test_build_source_file_catalog_deduplicates_and_preserves_order():
    rows = [
        {"source_file": "深圳市气象灾害应急预案.md"},
        {"source_file": "广东省自然灾害救助应急预案.md"},
        {"source_file": "深圳市气象灾害应急预案.md"},
    ]
    assert build_source_file_catalog(rows) == [
        "深圳市气象灾害应急预案.md",
        "广东省自然灾害救助应急预案.md",
    ]


def test_detect_source_file_filter_matches_filename_and_stem():
    source_files = [
        "深圳市气象灾害应急预案.md",
        "广东省自然灾害救助应急预案.md",
    ]
    assert detect_source_file_filter("深圳市气象灾害应急预案", source_files) == "深圳市气象灾害应急预案.md"
    assert detect_source_file_filter("请查看广东省自然灾害救助应急预案", source_files) == "广东省自然灾害救助应急预案.md"


def test_detect_metadata_filters_activates_only_source_file_and_keeps_reserved_hooks():
    source_files = [
        "深圳市气象灾害应急预案.md",
        "广东省自然灾害救助应急预案.md",
    ]
    plan = detect_metadata_filters("深圳市气象灾害应急预案中怎么规定", source_files)
    assert plan.source_file == "深圳市气象灾害应急预案.md"
    assert plan.active_filters == {"source_file": "深圳市气象灾害应急预案.md"}
    assert plan.region_hint == "深圳"
    assert plan.hazard_hint == "气象灾害"
    assert plan.document_type_hint == "预案"
    assert set(plan.reserved_fields.keys()) == {"region", "hazard", "document_type"}


def test_detect_metadata_filters_does_not_activate_non_source_file_filters():
    source_files = ["深圳市气象灾害应急预案.md"]
    plan = detect_metadata_filters("河南暴雨响应标准是什么", source_files)
    assert plan.source_file is None
    assert plan.active_filters == {}
    assert plan.region_hint == "河南"
    assert plan.hazard_hint == "暴雨"


def test_normalize_text_for_match_is_stable_for_chinese_punctuation():
    assert normalize_text_for_match("《深圳市气象灾害应急预案》") == "深圳市气象灾害应急预案"
