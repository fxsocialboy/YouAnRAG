"""CLI search entry for Stage3 hybrid retrieval + context packing."""

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
from rag_v2.retrieval.bm25_index import BM25Index
from rag_v2.retrieval.context_packer import ContextPacker
from rag_v2.retrieval.filters import build_source_file_catalog, detect_metadata_filters
from rag_v2.retrieval.hybrid_searcher import HybridSearcher
from rag_v2.retrieval.qdrant_searcher import QdrantSearcher


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--sparse-top-k", type=int, default=30)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cfg = get_config()
    metadata_rows = json.loads((cfg.stage1_artifacts_dir / "chunk_metadata.json").read_text(encoding="utf-8"))
    source_files = build_source_file_catalog(metadata_rows)
    filter_plan = detect_metadata_filters(args.query, source_files)
    dense = QdrantSearcher.from_config(cfg=cfg, device=args.device, batch_size=args.batch_size)
    sparse = BM25Index.load(cfg.artifacts_dir / "stage3" / "bm25_index.json")
    hybrid = HybridSearcher(dense_searcher=dense, sparse_index=sparse)
    packer = ContextPacker(metadata_rows)
    try:
        results = hybrid.search(
            args.query,
            top_k=args.top_k,
            dense_top_k=args.dense_top_k,
            sparse_top_k=args.sparse_top_k,
            filters=filter_plan.active_filters,
        )
        packed = packer.pack(results, token_budget=args.token_budget)
    finally:
        dense.close()

    payload = {
        "query": args.query,
        "filter_plan": filter_plan.to_dict(),
        "results": [item.to_dict() for item in results],
        "packed_context": packed.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
