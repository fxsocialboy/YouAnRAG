"""Build stage-1 chunks and metadata artifacts.

This script reads Markdown from the legacy snapshot, but writes only to
``artifacts/stage1``.  It does not modify legacy FAISS/index/metadata files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.config import get_config
from rag_v2.ingestion.chunker import build_chunks
from rag_v2.ingestion.markdown_parser import parse_markdown_file
from rag_v2.ingestion.metadata import build_quality_report, enrich_chunk_metadata, write_json, write_jsonl
from rag_v2.ingestion.token_counter import RegexTokenCounter


def build_stage1_chunks(markdown_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    cfg = get_config()
    cfg.validate(require_model=False)
    params = cfg.chunk_params
    source_dir = markdown_dir or cfg.source_markdown_dir
    target_dir = out_dir or cfg.stage1_artifacts_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    all_blocks = []
    files = sorted(source_dir.glob("*.md"))
    for path in files:
        all_blocks.extend(parse_markdown_file(path))

    chunks = build_chunks(all_blocks, params=params, token_counter=RegexTokenCounter())
    metadata_rows = [enrich_chunk_metadata(chunk) for chunk in chunks]
    report = build_quality_report(chunks, params).to_dict()
    report.update(
        {
            "source_markdown_dir": str(source_dir),
            "markdown_files": len(files),
            "input_blocks": len(all_blocks),
            "indexable_blocks": sum(1 for block in all_blocks if block.is_indexable),
            "chunk_params": params.__dict__,
        }
    )

    write_jsonl(target_dir / "chunks.jsonl", metadata_rows)
    write_json(target_dir / "chunk_metadata.json", metadata_rows)
    write_json(target_dir / "chunk_quality_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    report = build_stage1_chunks(args.markdown_dir, args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
