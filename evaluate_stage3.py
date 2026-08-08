"""Evaluate Stage3 dense/sparse/hybrid retrieval and context packing."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.config import get_config
from rag_v2.retrieval.bm25_index import BM25Index
from rag_v2.retrieval.context_packer import ContextPacker
from rag_v2.retrieval.filters import build_source_file_catalog, detect_metadata_filters
from rag_v2.retrieval.hybrid_searcher import HybridSearcher
from rag_v2.retrieval.qdrant_searcher import QdrantSearcher


def load_queries(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def doc_hit(items: list[dict[str, Any]], target_file: str, top_k: int) -> bool:
    return any(str(item.get("source_file", "")) == target_file for item in items[:top_k])


def chunk_hit(items: list[dict[str, Any]], target_file: str, target_chunk: int, top_k: int) -> bool:
    return any(
        str(item.get("source_file", "")) == target_file and int(item.get("chunk_index", -1)) == target_chunk
        for item in items[:top_k]
    )


def safe_mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def evaluate_queries(
    queries: list[dict[str, Any]],
    *,
    dense: QdrantSearcher,
    sparse: BM25Index,
    hybrid: HybridSearcher,
    packer: ContextPacker,
    source_files: list[str],
    top_k: int,
    dense_top_k: int,
    sparse_top_k: int,
    token_budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    dense_doc_hits = sparse_doc_hits = hybrid_doc_hits = 0
    dense_chunk_hits = sparse_chunk_hits = hybrid_chunk_hits = 0
    filter_hits = packed_nonempty = 0
    packed_token_ratios: list[float] = []
    latencies: list[float] = []

    for q in queries:
        target_file = q["relevant_source_file"]
        target_chunk = q["relevant_chunk_index"]
        plan = detect_metadata_filters(q["query"], source_files)

        t0 = time.perf_counter()
        dense_items = [item.to_dict() for item in dense.search(q["query"], top_k=top_k, filters=plan.active_filters)]
        sparse_items = [item.to_dict() for item in sparse.search(q["query"], top_k=top_k, source_file=plan.source_file)]
        hybrid_results = hybrid.search(
            q["query"],
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            filters=plan.active_filters,
        )
        hybrid_items = [item.to_dict() for item in hybrid_results]
        packed = packer.pack(hybrid_results, token_budget=token_budget).to_dict()
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(latency_ms)

        dense_doc = doc_hit(dense_items, target_file, top_k)
        sparse_doc = doc_hit(sparse_items, target_file, top_k)
        hybrid_doc = doc_hit(hybrid_items, target_file, top_k)
        dense_chunk = chunk_hit(dense_items, target_file, target_chunk, top_k)
        sparse_chunk = chunk_hit(sparse_items, target_file, target_chunk, top_k)
        hybrid_chunk = chunk_hit(hybrid_items, target_file, target_chunk, top_k)

        dense_doc_hits += int(dense_doc)
        sparse_doc_hits += int(sparse_doc)
        hybrid_doc_hits += int(hybrid_doc)
        dense_chunk_hits += int(dense_chunk)
        sparse_chunk_hits += int(sparse_chunk)
        hybrid_chunk_hits += int(hybrid_chunk)
        filter_hits += int((not plan.active_filters) or all(item.get("source_file") == target_file for item in hybrid_items))
        packed_nonempty += int(len(packed["evidence_chunks"]) > 0)
        packed_token_ratios.append(packed["total_tokens"] / max(token_budget, 1))

        rows.append(
            {
                **q,
                "filter_plan": plan.to_dict(),
                "latency_ms": latency_ms,
                "dense_doc_hit@10": dense_doc,
                "sparse_doc_hit@10": sparse_doc,
                "hybrid_doc_hit@10": hybrid_doc,
                "dense_chunk_hit_exact@10": dense_chunk,
                "sparse_chunk_hit_exact@10": sparse_chunk,
                "hybrid_chunk_hit_exact@10": hybrid_chunk,
                "dense_top10": dense_items,
                "sparse_top10": sparse_items,
                "hybrid_top10": hybrid_items,
                "packed_context": packed,
            }
        )
        print(
            f"{q['id']} dense={int(dense_doc)} sparse={int(sparse_doc)} hybrid={int(hybrid_doc)} "
            f"packed={len(packed['evidence_chunks'])} latency_ms={latency_ms}",
            flush=True,
        )

    n = len(queries) or 1
    summary = {
        "query_count": len(queries),
        "doc_recall@10": {
            "dense": round(dense_doc_hits / n, 4),
            "sparse": round(sparse_doc_hits / n, 4),
            "hybrid": round(hybrid_doc_hits / n, 4),
        },
        "chunk_recall_exact@10": {
            "dense": round(dense_chunk_hits / n, 4),
            "sparse": round(sparse_chunk_hits / n, 4),
            "hybrid": round(hybrid_chunk_hits / n, 4),
        },
        "hybrid_minus_dense_doc_recall@10": round((hybrid_doc_hits - dense_doc_hits) / n, 4),
        "filter_pass_ratio": round(filter_hits / n, 4),
        "packed_nonempty_ratio": round(packed_nonempty / n, 4),
        "avg_packed_token_ratio": safe_mean(packed_token_ratios),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
    }
    return summary, rows


def evaluate_stage3(
    *,
    natural_queries_path: Path,
    keyword_queries_path: Path,
    out_path: Path,
    top_k: int = 10,
    dense_top_k: int = 30,
    sparse_top_k: int = 30,
    token_budget: int = 1200,
    device: str = "cpu",
    batch_size: int = 16,
) -> dict[str, Any]:
    cfg = get_config()
    metadata_rows = json.loads((cfg.stage1_artifacts_dir / "chunk_metadata.json").read_text(encoding="utf-8"))
    source_files = build_source_file_catalog(metadata_rows)
    natural_queries = load_queries(natural_queries_path)
    keyword_queries = load_queries(keyword_queries_path)
    dense = QdrantSearcher.from_config(cfg=cfg, device=device, batch_size=batch_size)
    sparse = BM25Index.load(cfg.artifacts_dir / "stage3" / "bm25_index.json")
    hybrid = HybridSearcher(dense_searcher=dense, sparse_index=sparse)
    packer = ContextPacker(metadata_rows)
    started = time.perf_counter()
    try:
        natural_summary, natural_rows = evaluate_queries(
            natural_queries,
            dense=dense,
            sparse=sparse,
            hybrid=hybrid,
            packer=packer,
            source_files=source_files,
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            token_budget=token_budget,
        )
        keyword_summary, keyword_rows = evaluate_queries(
            keyword_queries,
            dense=dense,
            sparse=sparse,
            hybrid=hybrid,
            packer=packer,
            source_files=source_files,
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            token_budget=token_budget,
        )
    finally:
        dense.close()

    summary = {
        "top_k": top_k,
        "dense_top_k": dense_top_k,
        "sparse_top_k": sparse_top_k,
        "token_budget": token_budget,
        "natural_queries": natural_summary,
        "keyword_queries": keyword_summary,
        "stage3_gain_note": "Stage3 focuses on dense vs sparse vs hybrid retrieval behavior, especially on keyword-oriented queries.",
        "total_seconds": round(time.perf_counter() - started, 2),
    }
    payload = {
        "summary": summary,
        "natural_results": natural_rows,
        "keyword_results": keyword_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "experiments" / "eval_queries.jsonl")
    parser.add_argument("--keyword-queries", type=Path, default=PROJECT_ROOT / "experiments" / "eval_queries_keyword.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments" / "stage3_hybrid_eval.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--sparse-top-k", type=int, default=30)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = evaluate_stage3(
        natural_queries_path=args.queries,
        keyword_queries_path=args.keyword_queries,
        out_path=args.out,
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        token_budget=args.token_budget,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
