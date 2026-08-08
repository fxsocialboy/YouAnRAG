import json

import pytest

from rag_v2.schemas import Chunk, RetrievedChunk


def make_chunk(**overrides):
    data = {
        "chunk_id": "深圳市气象灾害应急预案.md::0",
        "source_file": "深圳市气象灾害应急预案.md",
        "chunk_index": 0,
        "section_path": ["深圳市气象灾害应急预案", "4 应急响应"],
        "content": "气象部门加强监测预报。",
        "embedding_text": "[文档] 深圳市气象灾害应急预案\n[章节] 4 应急响应\n[正文] 气象部门加强监测预报。",
        "token_count": 32,
        "char_count": 12,
        "is_indexable": True,
        "metadata": {"doc_type": "emergency_plan"},
    }
    data.update(overrides)
    return Chunk(**data)


def test_chunk_can_serialize_to_json_dict():
    chunk = make_chunk()
    as_dict = chunk.to_dict()
    assert as_dict["source_file"] == "深圳市气象灾害应急预案.md"
    assert as_dict["section_path"] == ["深圳市气象灾害应急预案", "4 应急响应"]
    encoded = json.dumps(as_dict, ensure_ascii=False)
    assert "深圳市气象灾害应急预案" in encoded


def test_chunk_from_dict_roundtrip():
    chunk = make_chunk()
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored == chunk


def test_chunk_normalizes_empty_section_items():
    chunk = make_chunk(section_path=["文档", "", "  ", "章节"])
    assert chunk.section_path == ["文档", "章节"]


@pytest.mark.parametrize(
    "overrides,error_type",
    [
        ({"chunk_id": ""}, ValueError),
        ({"source_file": ""}, ValueError),
        ({"chunk_index": -1}, ValueError),
        ({"token_count": -1}, ValueError),
        ({"char_count": -1}, ValueError),
        ({"section_path": "not-a-list"}, TypeError),
    ],
)
def test_invalid_chunk_rejected(overrides, error_type):
    with pytest.raises(error_type):
        make_chunk(**overrides)


def test_retrieved_chunk_legacy_adapter_returns_content():
    chunk = make_chunk(content="用于 legacy invoke 的原文。")
    result = RetrievedChunk(chunk=chunk, score=0.87, rank=1)
    assert result.to_legacy_content() == "用于 legacy invoke 的原文。"
    as_dict = result.to_dict()
    assert as_dict["score"] == 0.87
    assert as_dict["rank"] == 1
    assert as_dict["chunk"]["chunk_id"] == chunk.chunk_id
