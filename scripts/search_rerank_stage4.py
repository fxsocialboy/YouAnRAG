"""CLI search entry for Stage4 hybrid + rerank + MMR retrieval."""

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
from rag_v2.retrieval.mmr import MMRSelector
from rag_v2.retrieval.qdrant_searcher import QdrantSearcher
from rag_v2.retrieval.rerank_pipeline import RerankPipeline, RerankPipelineOptions
from rag_v2.retrieval.reranker import CrossEncoderReranker, FakeReranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--sparse-top-k", type=int, default=30)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--device", default="cpu", help="Embedding device used by Qdrant searcher and MMR passage embedding.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=True)
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("--mmr", dest="mmr", action="store_true", default=True)
    parser.add_argument("--no-mmr", dest="mmr", action="store_false")
    parser.add_argument("--fake-reranker", action="store_true", help="Use deterministic FakeReranker instead of loading a CrossEncoder model.")
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--mmr-pre-candidates", type=int, default=20)
    parser.add_argument("--mmr-top-k", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=30)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=None)
    parser.add_argument("--reranker-max-length", type=int, default=None)
    parser.add_argument("--reranker-model-path", type=Path, default=None)
    parser.add_argument("--reranker-model-name", default=None)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    cfg = get_config()
    metadata_rows = json.loads((cfg.stage1_artifacts_dir / "chunk_metadata.json").read_text(encoding="utf-8"))
    source_files = build_source_file_catalog(metadata_rows)
    filter_plan = detect_metadata_filters(args.query, source_files)

    dense = QdrantSearcher.from_config(cfg=cfg, device=args.device, batch_size=args.batch_size)
    sparse = BM25Index.load(cfg.artifacts_dir / "stage3" / "bm25_index.json")
    hybrid = HybridSearcher(dense_searcher=dense, sparse_index=sparse)
    reranker = None
    if args.rerank:
        if args.fake_reranker:
            reranker = FakeReranker(score_fn=lambda _query, candidate: candidate.fusion_score)
        else:
            model_ref = args.reranker_model_path or cfg.reranker_model_path or args.reranker_model_name or cfg.reranker_model_name
            reranker = CrossEncoderReranker(
                str(model_ref),
                device=args.reranker_device or cfg.reranker_device,
                batch_size=args.reranker_batch_size or cfg.reranker_batch_size,
                max_length=args.reranker_max_length or cfg.reranker_max_length,
            )
    mmr_selector = MMRSelector(dense.embedder, lambda_=args.mmr_lambda) if args.mmr else None
    pipeline = RerankPipeline.from_components(hybrid_searcher=hybrid, reranker=reranker, mmr_selector=mmr_selector)
    packer = ContextPacker(metadata_rows)
    options = RerankPipelineOptions(
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        rerank_top_k=args.rerank_top_k,
        mmr_pre_candidates=args.mmr_pre_candidates,
        mmr_top_k=args.mmr_top_k or args.top_k,
        enable_reranker=args.rerank,
        enable_mmr=args.mmr,
    )

    try:
        results = pipeline.search(args.query, filters=filter_plan.active_filters, options=options)
        packed = packer.pack(results, token_budget=args.token_budget)
    finally:
        dense.close()
        close = getattr(reranker, "close", None)
        if callable(close):
            close()

    payload = {
        "query": args.query,
        "filter_plan": filter_plan.to_dict(),
        "stage4_options": {
            **options.to_dict(),
            "fake_reranker": args.fake_reranker,
            "reranker_enabled": args.rerank,
            "mmr_enabled": args.mmr,
            "mmr_lambda": args.mmr_lambda,
        },
        "results": [item.to_dict() for item in results],
        "packed_context": packed.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
