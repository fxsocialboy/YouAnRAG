"""Evaluate stage-1 retrieval against the same natural-language query set.

Note about chunk metrics:
The current qrels were created before stage-1 re-chunking and contain legacy
``chunk_index`` values.  Therefore document recall is the primary comparable
metric.  ``chunk_recall_exact_index`` is still reported as a strict diagnostic,
but it is not a fair measure across different chunking strategies.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.retrieval.stage1_searcher import Stage1Searcher


def load_queries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_stage1(
    queries_path: Path,
    out_path: Path,
    top_k: int = 10,
    device: str = "cpu",
    batch_size: int = 16,
    limit: int = 0,
) -> dict:
    queries = load_queries(queries_path)
    if limit > 0:
        queries = queries[:limit]

    searcher = Stage1Searcher.from_config(device=device, batch_size=batch_size)
    results = []
    doc_hit_5 = doc_hit_10 = chunk_hit_5 = chunk_hit_10 = 0
    latencies = []
    started = time.perf_counter()

    for q in queries:
        t0 = time.perf_counter()
        hits = searcher.search(q["query"], top_k=top_k)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(latency_ms)
        target_file = q["relevant_source_file"]
        target_chunk = q["relevant_chunk_index"]
        doc5 = any(hit.source_file == target_file for hit in hits[:5])
        doc10 = any(hit.source_file == target_file for hit in hits[:10])
        chunk5 = any(hit.source_file == target_file and hit.chunk_index == target_chunk for hit in hits[:5])
        chunk10 = any(hit.source_file == target_file and hit.chunk_index == target_chunk for hit in hits[:10])
        doc_hit_5 += int(doc5)
        doc_hit_10 += int(doc10)
        chunk_hit_5 += int(chunk5)
        chunk_hit_10 += int(chunk10)
        row = {
            **q,
            "latency_ms": latency_ms,
            "doc_hit@5": doc5,
            "doc_hit@10": doc10,
            "chunk_hit_exact_index@5": chunk5,
            "chunk_hit_exact_index@10": chunk10,
            "top10": [hit.to_dict() for hit in hits[:10]],
        }
        results.append(row)
        print(f"{q['id']} doc@10={int(doc10)} chunk_exact@10={int(chunk10)} latency_ms={latency_ms}", flush=True)

    n = len(queries) or 1
    summary = {
        "query_count": len(queries),
        "top_k": top_k,
        "doc_recall@5": round(doc_hit_5 / n, 4),
        "doc_recall@10": round(doc_hit_10 / n, 4),
        "chunk_recall_exact_index@5": round(chunk_hit_5 / n, 4),
        "chunk_recall_exact_index@10": round(chunk_hit_10 / n, 4),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2) if latencies else 0,
        "total_seconds": round(time.perf_counter() - started, 2),
        "metric_note": "chunk_recall_exact_index uses legacy qrels and is strict diagnostic only after re-chunking; doc_recall is the primary comparable metric.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary, "results": results}


def write_comparison_report(stage1_result: dict, legacy_result_path: Path, out_path: Path) -> None:
    legacy = json.loads(legacy_result_path.read_text(encoding="utf-8")) if legacy_result_path.exists() else None
    s = stage1_result["summary"]
    lines = [
        "# Stage1 vs Legacy Retrieval Report",
        "",
        "## Stage1 Summary",
        "",
        "```json",
        json.dumps(s, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if legacy:
        l = legacy.get("summary", {})
        lines.extend(
            [
                "## Legacy Summary",
                "",
                "```json",
                json.dumps(l, ensure_ascii=False, indent=2),
                "```",
                "",
                "## Comparable Metrics",
                "",
                "| Metric | Legacy | Stage1 | Note |",
                "|---|---:|---:|---|",
                f"| doc_recall@10 | {l.get('doc_recall@10', 'N/A')} | {s.get('doc_recall@10')} | primary comparable metric |",
                f"| chunk exact recall@10 | {l.get('chunk_recall@10', l.get('chunk_recall_exact_index@10', 'N/A'))} | {s.get('chunk_recall_exact_index@10')} | strict; qrels use legacy chunk_index |",
                f"| avg/p95 latency | N/A | {s.get('avg_latency_ms')} / {s.get('p95_latency_ms')} ms | includes query embedding |",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Stage1 uses a new chunking strategy, so legacy `chunk_index` labels are not directly equivalent to Stage1 `chunk_index`. ",
            "For this phase, use document-level recall and qualitative Top-10 inspection as the main comparison. ",
            "If strict chunk-level evaluation is needed later, create new qrels against Stage1 chunk IDs.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "experiments" / "eval_queries.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments" / "stage1_baseline_results.json")
    parser.add_argument("--legacy", type=Path, default=PROJECT_ROOT / "experiments" / "legacy_baseline_results.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "experiments" / "stage1_vs_legacy_report.md")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    result = evaluate_stage1(args.queries, args.out, args.top_k, args.device, args.batch_size, args.limit)
    write_comparison_report(result, args.legacy, args.report)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
