"""CLI search entry for Stage2 Qdrant retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.retrieval.qdrant_searcher import QdrantSearcher


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--source-file", default=None)
    args = parser.parse_args()

    filters = {"source_file": args.source_file} if args.source_file else None
    searcher = QdrantSearcher.from_config(device=args.device, batch_size=args.batch_size)
    try:
        results = searcher.search_dicts(args.query, top_k=args.top_k, filters=filters)
    finally:
        searcher.close()
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
