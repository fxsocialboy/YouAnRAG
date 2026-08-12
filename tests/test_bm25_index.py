from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from rag_v2.retrieval.bm25_index import BM25Index, tokenize_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_bm25_stage3.py"

spec = importlib.util.spec_from_file_location("build_bm25_stage3", SCRIPT_PATH)
build_bm25_stage3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(build_bm25_stage3)


def sample_rows() -> list[dict]:
    return [
        {
            "chunk_id": "weather.md::0",
            "source_file": "weather.md",
            "chunk_index": 0,
            "section_path_text": "IV级响应",
            "content": "IV级气象灾害应急响应由办公室启动，24小时降雨量100mm时重点关注。",
            "token_count": 30,
            "metadata": {"section_path_text": "IV级响应"},
        },
        {
            "chunk_id": "flood.md::0",
            "source_file": "flood.md",
            "chunk_index": 0,
            "section_path_text": "暴雨预警",
            "content": "暴雨蓝色预警时学校停课安排和转移流程。",
            "token_count": 20,
            "metadata": {"section_path_text": "暴雨预警"},
        },
        {
            "chunk_id": "guangdong_typhoon.md::0",
            "source_file": "guangdong_typhoon.md",
            "chunk_index": 0,
            "section_path_text": "台风条款",
            "content": "广东省台风应急预案规定，台风黄色预警需要学校做好停课准备。",
            "token_count": 24,
            "metadata": {"section_path_text": "台风条款"},
        },
    ]


def test_tokenize_text_keeps_ascii_digits_and_cjk():
    tokens = tokenize_text("IV级响应 24小时 100mm 广东台风")
    assert "iv" in tokens
    assert "24" in tokens
    assert "100mm" in tokens or "100" in tokens
    assert "广" in tokens
    assert "东" in tokens


def test_bm25_search_hits_keyword_and_file_name():
    index = BM25Index.from_chunk_metadata(sample_rows())
    hits = index.search("IV级响应 启动 100mm", top_k=3)
    assert hits[0].chunk_id == "weather.md::0"
    assert "100mm" in hits[0].content

    file_hits = index.search("guangdong_typhoon", top_k=3)
    assert file_hits[0].source_file == "guangdong_typhoon.md"


def test_bm25_source_file_filter_and_empty_query():
    index = BM25Index.from_chunk_metadata(sample_rows())
    hits = index.search("台风 黄色预警", top_k=3, source_file="guangdong_typhoon.md")
    assert [hit.source_file for hit in hits] == ["guangdong_typhoon.md"]
    assert index.search("", top_k=3) == []


def test_build_bm25_script_creates_index_and_report(tmp_path: Path):
    metadata_path = tmp_path / "chunk_metadata.json"
    out_path = tmp_path / "bm25_index.json"
    report_path = tmp_path / "bm25_build_report.json"
    metadata_path.write_text(json.dumps(sample_rows(), ensure_ascii=False), encoding="utf-8")

    report = build_bm25_stage3.build_bm25_index(metadata_path, out_path, report_path)
    assert out_path.exists()
    assert report_path.exists()
    assert report["doc_count"] == 3

    loaded = BM25Index.load(out_path)
    hits = loaded.search("台风 黄色预警", top_k=2)
    assert hits[0].chunk_id == "guangdong_typhoon.md::0"
