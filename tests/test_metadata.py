import json
from pathlib import Path

from rag_v2.config import Stage1ChunkParams
from rag_v2.ingestion.metadata import (
    build_quality_report,
    count_duplicate_contents,
    enrich_chunk_metadata,
    sha256_text,
    write_json,
    write_jsonl,
)
from rag_v2.schemas import Chunk


def make_chunk(**overrides):
    data = {
        "chunk_id": "doc.md::0",
        "source_file": "doc.md",
        "chunk_index": 0,
        "section_path": ["doc", "section"],
        "content": "正文内容",
        "embedding_text": "[文档] doc\n[章节] doc > section\n[正文] 正文内容",
        "token_count": 20,
        "char_count": 4,
        "is_indexable": True,
        "metadata": {},
    }
    data.update(overrides)
    return Chunk(**data)


def test_enrich_chunk_metadata_adds_hash_and_section_text():
    chunk = make_chunk()
    row = enrich_chunk_metadata(chunk)
    assert row["chunk_id"] == "doc.md::0"
    assert row["section_path_text"] == "doc > section"
    assert row["content_hash"] == sha256_text("正文内容")
    assert row["embedding_text_hash"] == sha256_text(chunk.embedding_text)
    assert row["content"] == "正文内容"
    assert row["embedding_text"].startswith("[文档]")


def test_count_duplicate_contents_counts_extra_duplicates_only():
    chunks = [
        make_chunk(chunk_id="a::0", content="重复", chunk_index=0),
        make_chunk(chunk_id="a::1", content="重复", chunk_index=1),
        make_chunk(chunk_id="a::2", content="不同", chunk_index=2),
    ]
    assert count_duplicate_contents(chunks) == 1


def test_build_quality_report_core_metrics():
    params = Stage1ChunkParams(hard_max_tokens=40)
    chunks = [
        make_chunk(chunk_id="a.md::0", source_file="a.md", chunk_index=0, token_count=20, char_count=10),
        make_chunk(chunk_id="a.md::1", source_file="a.md", chunk_index=1, token_count=45, char_count=12, content="重复"),
        make_chunk(chunk_id="b.md::0", source_file="b.md", chunk_index=0, token_count=30, char_count=15, content="重复", section_path=[]),
    ]
    report = build_quality_report(chunks, params)
    assert report.total_chunks == 3
    assert report.indexable_chunks == 3
    assert report.unique_source_files == 2
    assert report.max_tokens == 45
    assert report.over_hard_max_chunks == 1
    assert report.missing_section_path_chunks == 1
    assert report.duplicate_chunks == 1
    assert report.to_dict()["hard_max_tokens"] == 40


def test_write_jsonl_and_json():
    # Use project-controlled path if pytest tmp_path is unavailable in some
    # Windows sandbox setups.  This test still accepts tmp_path when it works.
    base = Path(r"G:\tiaozhanbei\newrag\artifacts\stage1\test_tmp_metadata")
    base.mkdir(parents=True, exist_ok=True)
    jsonl_path = base / "rows.jsonl"
    json_path = base / "data.json"
    try:
        write_jsonl(jsonl_path, [{"a": 1}, {"b": "中文"}])
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        assert rows == [{"a": 1}, {"b": "中文"}]
        write_json(json_path, {"ok": True, "text": "中文"})
        assert json.loads(json_path.read_text(encoding="utf-8"))["text"] == "中文"
    finally:
        for p in [jsonl_path, json_path]:
            if p.exists():
                p.unlink()
