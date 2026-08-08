import json
from pathlib import Path

from scripts.build_chunks_stage1 import build_stage1_chunks


def test_build_stage1_chunks_writes_expected_artifacts():
    base = Path(r"G:\tiaozhanbei\newrag\artifacts\stage1\test_build_chunks")
    md_dir = base / "mds"
    out_dir = base / "out"
    md_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = md_dir / "测试预案.md"
    try:
        source.write_text(
            "# 测试预案\n\n## 1 应急响应\n\n气象部门加强监测预报。\n\n参考文献：\n[1] Zhang A. Journal of Disaster.",
            encoding="utf-8",
        )
        report = build_stage1_chunks(markdown_dir=md_dir, out_dir=out_dir)
        assert report["markdown_files"] == 1
        assert report["total_chunks"] >= 1
        assert report["over_hard_max_chunks"] == 0
        assert (out_dir / "chunks.jsonl").exists()
        assert (out_dir / "chunk_metadata.json").exists()
        assert (out_dir / "chunk_quality_report.json").exists()
        metadata = json.loads((out_dir / "chunk_metadata.json").read_text(encoding="utf-8"))
        assert metadata[0]["source_file"] == "测试预案.md"
        assert metadata[0]["section_path"] == ["测试预案", "1 应急响应"]
        assert "[章节]" in metadata[0]["embedding_text"]
        assert "content_hash" in metadata[0]
    finally:
        for child in sorted(base.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
