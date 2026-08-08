"""Command-line search over the stage-1 optimized FAISS index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.retrieval.stage1_searcher import Stage1Searcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    args = parser.parse_args()

    query = args.query or input("请输入查询语句：")
    searcher = Stage1Searcher.from_config(device=args.device, batch_size=args.batch_size)
    results = searcher.search(query, top_k=args.top_k)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        return

    print(f"\nStage1 查询结果 Top {args.top_k}:\n")
    for result in results:
        print(f"Top {result.rank} | L2距离: {result.score_l2:.4f}")
        print(f"文件: {result.source_file} | Chunk #{result.chunk_index} | Global #{result.global_index}")
        print(f"章节: {result.section_path_text}")
        print(f"内容: {result.content_preview}{'...' if len(result.content) > len(result.content_preview) else ''}")
        print("-" * 80)


if __name__ == "__main__":
    main()
