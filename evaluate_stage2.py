"""Evaluate Stage2 Qdrant retrieval by migration consistency against Stage1 FAISS."""

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
from rag_v2.retrieval.qdrant_searcher import QdrantSearcher
from rag_v2.retrieval.stage1_searcher import Stage1Searcher
from rag_v2.sync.registry import DocumentRegistry


REQUIRED_PAYLOAD_FIELDS = [
    "chunk_id",
    "source_file",
    "chunk_index",
    "section_path_text",
    "content",
    "content_hash",
    "token_count",
    "is_active",
]


def load_queries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_ids(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("chunk_id", "")) for item in items]


def prefix_match_ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for x, y in zip(a[:n], b[:n]) if x == y) / n


def set_overlap_ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    denom = max(len(a), len(b), 1)
    return len(set(a) & set(b)) / denom


def payload_completeness(top_hits: list[dict[str, Any]]) -> float:
    if not top_hits:
        return 1.0
    total = len(top_hits) * len(REQUIRED_PAYLOAD_FIELDS)
    ok = 0
    for item in top_hits:
        for field in REQUIRED_PAYLOAD_FIELDS:
            value = item.get(field)
            if value is not None and value != "":
                ok += 1
    return ok / total


def evaluate_stage2(
    queries_path: Path,
    out_path: Path,
    top_k: int = 10,
    device: str = "cpu",
    batch_size: int = 16,
    limit: int = 0,
) -> dict[str, Any]:
    cfg = get_config()
    queries = load_queries(queries_path)
    if limit > 0:
        queries = queries[:limit]

    stage1 = Stage1Searcher.from_config(cfg=cfg, device=device, batch_size=batch_size)
    stage2 = QdrantSearcher.from_config(cfg=cfg, device=device, batch_size=batch_size)

    rows = []
    prefix_scores = []
    set_scores = []
    payload_scores = []
    filter_ok = 0
    latencies = []
    started = time.perf_counter()
    point_count = stage2.store.count()
    try:
        for q in queries:
            t0 = time.perf_counter()
            s1 = [item.to_dict() for item in stage1.search(q["query"], top_k=top_k)]
            s2 = [item.to_dict() for item in stage2.search(q["query"], top_k=top_k)]
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            latencies.append(latency_ms)

            ids1 = chunk_ids(s1)
            ids2 = chunk_ids(s2)
            prefix_ratio = prefix_match_ratio(ids1, ids2)
            set_ratio = set_overlap_ratio(ids1, ids2)
            payload_ratio = payload_completeness(s2)
            prefix_scores.append(prefix_ratio)
            set_scores.append(set_ratio)
            payload_scores.append(payload_ratio)

            target_file = q["relevant_source_file"]
            filtered = [item.to_dict() for item in stage2.search(q["query"], top_k=top_k, filters={"source_file": target_file})]
            filter_pass = bool(filtered) and all(item.get("source_file") == target_file for item in filtered)
            filter_ok += int(filter_pass)

            rows.append({
                **q,
                "latency_ms": latency_ms,
                "stage1_top10": s1,
                "stage2_top10": s2,
                "stage1_top10_chunk_ids": ids1,
                "stage2_top10_chunk_ids": ids2,
                "top10_prefix_match_ratio": round(prefix_ratio, 4),
                "top10_set_overlap_ratio": round(set_ratio, 4),
                "payload_completeness_ratio": round(payload_ratio, 4),
                "filter_target_source_file": target_file,
                "filter_pass": filter_pass,
                "filtered_top10": filtered,
            })
            print(f"{q['id']} prefix={prefix_ratio:.2f} overlap={set_ratio:.2f} filter={int(filter_pass)} latency_ms={latency_ms}", flush=True)
    finally:
        stage1_close = getattr(stage1, 'close', None)
        if callable(stage1_close):
            stage1_close()
        stage2.close()

    registry = DocumentRegistry(cfg.registry_db_path)
    metadata_count = len(json.loads((cfg.stage1_artifacts_dir / "chunk_metadata.json").read_text(encoding="utf-8")))
    active_registry_document_count = len(registry.list_documents(include_deleted=False)) if cfg.registry_db_path.exists() else 0
    deleted_registry_document_count = sum(1 for item in registry.list_documents(include_deleted=True) if item.status == "deleted") if cfg.registry_db_path.exists() else 0

    summary = {
        "query_count": len(queries),
        "top_k": top_k,
        "avg_top10_prefix_match_ratio": round(statistics.mean(prefix_scores), 4) if prefix_scores else 0,
        "avg_top10_set_overlap_ratio": round(statistics.mean(set_scores), 4) if set_scores else 0,
        "min_top10_set_overlap_ratio": round(min(set_scores), 4) if set_scores else 0,
        "payload_completeness_ratio": round(statistics.mean(payload_scores), 4) if payload_scores else 0,
        "filter_pass_ratio": round(filter_ok / (len(queries) or 1), 4),
        "qdrant_point_count": point_count,
        "metadata_count": metadata_count,
        "point_count_matches_metadata": point_count == metadata_count,
        "active_registry_document_count": active_registry_document_count,
        "deleted_registry_document_count": deleted_registry_document_count,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2) if latencies else 0,
        "total_seconds": round(time.perf_counter() - started, 2),
        "metric_note": "Stage2 focuses on migration consistency between Stage1 FAISS and Qdrant, not recall lift.",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary, "results": rows}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "experiments" / "eval_queries.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments" / "stage2_consistency_results.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    result = evaluate_stage2(args.queries, args.out, args.top_k, args.device, args.batch_size, args.limit)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
