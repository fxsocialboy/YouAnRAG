"""Evaluate Stage5 query planning / rewrite / multi-query / HyDE variants."""

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
from rag_v2.query.analyzer import QueryAnalyzer
from rag_v2.query.hyde import DeepSeekHydeGenerator, DisabledHydeGenerator, FakeHydeGenerator, RuleBasedHydeGenerator
from rag_v2.query.rewriter import QueryRewriter
from rag_v2.retrieval.bm25_index import BM25Index
from rag_v2.retrieval.context_packer import ContextPacker
from rag_v2.retrieval.filters import build_source_file_catalog, detect_metadata_filters
from rag_v2.retrieval.hybrid_searcher import HybridSearcher
from rag_v2.retrieval.mmr import MMRSelector
from rag_v2.retrieval.multi_query_pipeline import MultiQueryPipeline, MultiQueryPipelineOptions
from rag_v2.retrieval.qdrant_searcher import QdrantSearcher
from rag_v2.retrieval.rerank_pipeline import RerankPipeline, RerankPipelineOptions
from rag_v2.retrieval.reranker import CrossEncoderReranker, FakeReranker

VARIANTS = ["stage4_raw", "stage5_rewrite", "stage5_multi_query", "stage5_hyde"]
QUERY_GROUPS = ["scenario", "short_ambiguous", "exact_guardrail"]


class FakeMMRSelector:
    def select(self, candidates, top_k: int):
        selected = candidates[:top_k]
        for rank, candidate in enumerate(selected, 1):
            candidate.mmr_score = round(1.0 / rank, 6)
            candidate.rank = rank
            candidate.stage4_rank = rank
        return selected


def load_queries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def maybe_limit(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return items if limit is None or limit <= 0 else items[:limit]


def rank_of_doc(items: list[dict[str, Any]], target_file: str, top_k: int) -> int | None:
    for idx, item in enumerate(items[:top_k], 1):
        if str(item.get("source_file", "")) == target_file:
            return idx
    return None


def rank_of_chunk(items: list[dict[str, Any]], target_file: str, target_chunk: int, top_k: int) -> int | None:
    for idx, item in enumerate(items[:top_k], 1):
        if str(item.get("source_file", "")) == target_file and int(item.get("chunk_index", -1)) == target_chunk:
            return idx
    return None


def reciprocal_rank(rank: int | None) -> float:
    return round(1.0 / rank, 6) if rank else 0.0


def safe_mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def rank_shift(before: int | None, after: int | None, miss_rank: int) -> int | None:
    if before is None and after is None:
        return None
    return (after if after is not None else miss_rank) - (before if before is not None else miss_rank)


def summarize_rows(rows: list[dict[str, Any]], *, top_k: int, guardrail: bool = False) -> dict[str, Any]:
    n = len(rows) or 1
    summary: dict[str, Any] = {"query_count": len(rows), "variants": {}, "top_k": top_k}
    for variant in VARIANTS:
        doc_hits = chunk_hits = packed_nonempty = 0
        doc_mrrs: list[float] = []
        chunk_mrrs: list[float] = []
        latencies: list[float] = []
        token_ratios: list[float] = []
        branch_counts: list[float] = []
        hyde_used = fallback_count = 0
        doc_shifts: list[int] = []
        chunk_shifts: list[int] = []
        for row in rows:
            metrics = row["variant_metrics"][variant]
            doc_hits += int(metrics["doc_hit@10"])
            chunk_hits += int(metrics["chunk_hit_exact@10"])
            packed_nonempty += int(metrics["packed_nonempty"])
            doc_mrrs.append(float(metrics["doc_mrr@10"]))
            chunk_mrrs.append(float(metrics["chunk_mrr_exact@10"]))
            latencies.append(float(metrics["latency_ms"]))
            token_ratios.append(float(metrics["packed_token_ratio"]))
            branch_counts.append(float(metrics.get("branch_count", 1)))
            hyde_used += int(metrics.get("hyde_used", False))
            fallback_count += int(metrics.get("fallback", False))
            if metrics.get("doc_rank_shift_vs_stage4") is not None:
                doc_shifts.append(int(metrics["doc_rank_shift_vs_stage4"]))
            if metrics.get("chunk_rank_shift_vs_stage4") is not None:
                chunk_shifts.append(int(metrics["chunk_rank_shift_vs_stage4"]))
        summary["variants"][variant] = {
            "doc_recall@10": round(doc_hits / n, 4),
            "chunk_recall_exact@10": round(chunk_hits / n, 4),
            "doc_mrr@10": safe_mean(doc_mrrs),
            "chunk_mrr_exact@10": safe_mean(chunk_mrrs),
            "packed_nonempty_ratio": round(packed_nonempty / n, 4),
            "avg_packed_token_ratio": safe_mean(token_ratios),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "avg_branch_count": safe_mean(branch_counts),
            "hyde_used_ratio": round(hyde_used / n, 4),
            "fallback_count": fallback_count,
            "avg_doc_rank_shift_vs_stage4": safe_mean([float(x) for x in doc_shifts]),
            "avg_chunk_rank_shift_vs_stage4": safe_mean([float(x) for x in chunk_shifts]),
        }
    base = summary["variants"].get("stage4_raw", {})
    for variant, metrics in summary["variants"].items():
        metrics["doc_mrr_delta_vs_stage4"] = round(float(metrics["doc_mrr@10"]) - float(base.get("doc_mrr@10", 0.0)), 4)
        metrics["chunk_mrr_delta_vs_stage4"] = round(float(metrics["chunk_mrr_exact@10"]) - float(base.get("chunk_mrr_exact@10", 0.0)), 4)
        if guardrail and variant != "stage4_raw":
            metrics["guardrail_no_regression"] = (
                float(metrics["doc_recall@10"]) >= float(base.get("doc_recall@10", 0.0))
                and float(metrics["chunk_recall_exact@10"]) >= float(base.get("chunk_recall_exact@10", 0.0))
            )
    return summary


def evaluate_query_set(
    queries: list[dict[str, Any]],
    *,
    source_files: list[str],
    stage4: RerankPipeline,
    stage5_disabled: MultiQueryPipeline,
    stage5_hyde: MultiQueryPipeline,
    packer: ContextPacker,
    top_k: int,
    dense_top_k: int,
    sparse_top_k: int,
    rerank_top_k: int,
    mmr_pre_candidates: int,
    mmr_top_k: int,
    token_budget: int,
    guardrail: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for q in queries:
        filter_plan = detect_metadata_filters(q["query"], source_files)
        target_file = q["relevant_source_file"]
        target_chunk = int(q["relevant_chunk_index"])
        variant_results: dict[str, list[dict[str, Any]]] = {}
        variant_packs: dict[str, dict[str, Any]] = {}
        variant_metrics: dict[str, dict[str, Any]] = {}
        variant_plans: dict[str, Any] = {}

        stage4_options = RerankPipelineOptions(
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            rerank_top_k=rerank_top_k,
            mmr_pre_candidates=mmr_pre_candidates,
            mmr_top_k=mmr_top_k,
            enable_reranker=True,
            enable_mmr=True,
        )
        stage5_base_options = MultiQueryPipelineOptions(
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            rerank_top_k=rerank_top_k,
            mmr_pre_candidates=mmr_pre_candidates,
            mmr_top_k=mmr_top_k,
            branch_candidate_top_k=max(top_k, rerank_top_k),
        )
        variant_calls = {
            "stage4_raw": lambda: (stage4.search(q["query"], filters=filter_plan.active_filters, options=stage4_options), None),
            "stage5_rewrite": lambda: _stage5_call(stage5_disabled, q["query"], filter_plan.active_filters, stage5_base_options, max_branches=2),
            "stage5_multi_query": lambda: _stage5_call(stage5_disabled, q["query"], filter_plan.active_filters, stage5_base_options, max_branches=None),
            "stage5_hyde": lambda: _stage5_call(stage5_hyde, q["query"], filter_plan.active_filters, stage5_base_options, max_branches=None),
        }
        for variant in VARIANTS:
            t0 = time.perf_counter()
            results, output = variant_calls[variant]()
            packed = packer.pack(results, token_budget=token_budget)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            items = [item.to_dict() for item in results]
            packed_dict = packed.to_dict()
            doc_rank = rank_of_doc(items, target_file, top_k)
            chunk_rank = rank_of_chunk(items, target_file, target_chunk, top_k)
            variant_results[variant] = items
            variant_packs[variant] = packed_dict
            if output is not None:
                variant_plans[variant] = {
                    "query_plan": output.query_plan.to_dict(),
                    "query_branches": [b.to_dict() for b in output.query_branches],
                    "hyde_document": output.hyde_document.to_dict() if output.hyde_document else None,
                }
            hyde_doc = output.hyde_document if output is not None else None
            variant_metrics[variant] = {
                "latency_ms": latency_ms,
                "doc_rank@10": doc_rank,
                "chunk_rank_exact@10": chunk_rank,
                "doc_hit@10": doc_rank is not None,
                "chunk_hit_exact@10": chunk_rank is not None,
                "doc_mrr@10": reciprocal_rank(doc_rank),
                "chunk_mrr_exact@10": reciprocal_rank(chunk_rank),
                "packed_nonempty": len(packed_dict["evidence_chunks"]) > 0,
                "packed_token_ratio": round(packed_dict["total_tokens"] / max(token_budget, 1), 4),
                "branch_count": len(output.query_branches) if output is not None else 1,
                "hyde_used": bool(hyde_doc and hyde_doc.used),
                "fallback": bool(hyde_doc and "fallback" in hyde_doc.reason),
            }

        base_doc_rank = variant_metrics["stage4_raw"]["doc_rank@10"]
        base_chunk_rank = variant_metrics["stage4_raw"]["chunk_rank_exact@10"]
        miss_rank = top_k + 1
        for variant in VARIANTS:
            variant_metrics[variant]["doc_rank_shift_vs_stage4"] = rank_shift(base_doc_rank, variant_metrics[variant]["doc_rank@10"], miss_rank)
            variant_metrics[variant]["chunk_rank_shift_vs_stage4"] = rank_shift(base_chunk_rank, variant_metrics[variant]["chunk_rank_exact@10"], miss_rank)

        row = {
            **q,
            "filter_plan": filter_plan.to_dict(),
            "variant_metrics": variant_metrics,
            "variant_results": variant_results,
            "variant_packed_context": variant_packs,
            "variant_plans": variant_plans,
        }
        rows.append(row)
        print(f"{q['id']} " + " ".join(f"{v}:doc_mrr={variant_metrics[v]['doc_mrr@10']}" for v in VARIANTS), flush=True)
    return summarize_rows(rows, top_k=top_k, guardrail=guardrail), rows


def _stage5_call(pipeline: MultiQueryPipeline, query: str, filters: dict[str, Any], base_options: MultiQueryPipelineOptions, max_branches: int | None):
    options = MultiQueryPipelineOptions(**{**base_options.to_dict(), "max_branches": max_branches})
    output = pipeline.search(query, filters=filters, options=options)
    return output.results, output


def make_reranker(*, fake_reranker: bool, reranker_model_ref: str, device: str, batch_size: int, max_length: int):
    if fake_reranker:
        return FakeReranker(score_fn=lambda _query, candidate: candidate.fusion_score)
    return CrossEncoderReranker(reranker_model_ref, device=device, batch_size=batch_size, max_length=max_length)


def make_hyde_generator(mode: str, *, fake_hyde: bool, cfg):
    if fake_hyde:
        return FakeHydeGenerator()
    if mode == "disabled":
        return DisabledHydeGenerator()
    if mode == "rule":
        return RuleBasedHydeGenerator()
    if mode == "deepseek":
        return DeepSeekHydeGenerator(
            api_key=cfg.deepseek_api_key,
            model=cfg.deepseek_model,
            timeout=cfg.deepseek_timeout,
            max_retries=cfg.deepseek_max_retries,
            fallback=RuleBasedHydeGenerator(),
        )
    raise ValueError(f"unsupported hyde mode: {mode}")


def evaluate_stage5(
    *,
    scenario_queries_path: Path,
    short_queries_path: Path,
    guardrail_queries_path: Path,
    out_path: Path,
    top_k: int = 10,
    dense_top_k: int = 30,
    sparse_top_k: int = 30,
    rerank_top_k: int = 30,
    mmr_pre_candidates: int = 20,
    mmr_top_k: int = 10,
    token_budget: int = 1200,
    device: str = "cpu",
    batch_size: int = 16,
    reranker_device: str = "cpu",
    reranker_batch_size: int = 16,
    reranker_max_length: int = 512,
    reranker_model_ref: str | None = None,
    fake_reranker: bool = False,
    fake_mmr: bool = False,
    fake_hyde: bool = False,
    hyde_mode: str = "rule",
    mmr_lambda: float = 0.7,
    limit: int | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    metadata_rows = json.loads((cfg.stage1_artifacts_dir / "chunk_metadata.json").read_text(encoding="utf-8"))
    source_files = build_source_file_catalog(metadata_rows)
    scenario_queries = maybe_limit(load_queries(scenario_queries_path), limit)
    short_queries = maybe_limit(load_queries(short_queries_path), limit)
    guardrail_queries = maybe_limit(load_queries(guardrail_queries_path), limit)

    dense = QdrantSearcher.from_config(cfg=cfg, device=device, batch_size=batch_size)
    sparse = BM25Index.load(cfg.artifacts_dir / "stage3" / "bm25_index.json")
    hybrid = HybridSearcher(dense_searcher=dense, sparse_index=sparse)
    reranker_ref = reranker_model_ref or str(cfg.reranker_model_path or cfg.reranker_model_name)
    reranker = make_reranker(fake_reranker=fake_reranker, reranker_model_ref=reranker_ref, device=reranker_device, batch_size=reranker_batch_size, max_length=reranker_max_length)
    mmr_selector = FakeMMRSelector() if fake_mmr else MMRSelector(dense.embedder, lambda_=mmr_lambda)
    stage4 = RerankPipeline.from_components(hybrid_searcher=hybrid, reranker=reranker, mmr_selector=mmr_selector)
    stage5_disabled = MultiQueryPipeline(stage4, analyzer=QueryAnalyzer(), rewriter=QueryRewriter(), hyde_generator=DisabledHydeGenerator())
    stage5_hyde = MultiQueryPipeline(stage4, analyzer=QueryAnalyzer(), rewriter=QueryRewriter(), hyde_generator=make_hyde_generator(hyde_mode, fake_hyde=fake_hyde, cfg=cfg))
    packer = ContextPacker(metadata_rows)
    started = time.perf_counter()
    try:
        common = dict(source_files=source_files, stage4=stage4, stage5_disabled=stage5_disabled, stage5_hyde=stage5_hyde, packer=packer, top_k=top_k, dense_top_k=dense_top_k, sparse_top_k=sparse_top_k, rerank_top_k=rerank_top_k, mmr_pre_candidates=mmr_pre_candidates, mmr_top_k=mmr_top_k, token_budget=token_budget)
        scenario_summary, scenario_rows = evaluate_query_set(scenario_queries, **common)
        short_summary, short_rows = evaluate_query_set(short_queries, **common)
        guardrail_summary, guardrail_rows = evaluate_query_set(guardrail_queries, guardrail=True, **common)
    finally:
        dense.close()
        close = getattr(reranker, "close", None)
        if callable(close):
            close()

    summary = {
        "top_k": top_k,
        "dense_top_k": dense_top_k,
        "sparse_top_k": sparse_top_k,
        "rerank_top_k": rerank_top_k,
        "mmr_pre_candidates": mmr_pre_candidates,
        "mmr_top_k": mmr_top_k,
        "token_budget": token_budget,
        "fake_reranker": fake_reranker,
        "fake_mmr": fake_mmr,
        "fake_hyde": fake_hyde,
        "hyde_mode": hyde_mode,
        "mmr_lambda": mmr_lambda,
        "reranker_model_ref": reranker_ref,
        "scenario_queries": scenario_summary,
        "short_ambiguous_queries": short_summary,
        "exact_guardrail_queries": guardrail_summary,
        "total_seconds": round(time.perf_counter() - started, 2),
    }
    payload = {"summary": summary, "scenario_results": scenario_rows, "short_ambiguous_results": short_rows, "exact_guardrail_results": guardrail_rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-queries", type=Path, default=PROJECT_ROOT / "experiments" / "eval_queries_stage5_scenario.jsonl")
    parser.add_argument("--short-queries", type=Path, default=PROJECT_ROOT / "experiments" / "eval_queries_stage5_short_ambiguous.jsonl")
    parser.add_argument("--guardrail-queries", type=Path, default=PROJECT_ROOT / "experiments" / "eval_queries_stage5_exact_guardrail.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments" / "stage5_query_eval.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--sparse-top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=30)
    parser.add_argument("--mmr-pre-candidates", type=int, default=20)
    parser.add_argument("--mmr-top-k", type=int, default=10)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--reranker-model-path", type=Path, default=None)
    parser.add_argument("--reranker-model-name", default=None)
    parser.add_argument("--fake-reranker", action="store_true")
    parser.add_argument("--fake-mmr", action="store_true")
    parser.add_argument("--fake-hyde", action="store_true")
    parser.add_argument("--hyde-mode", choices=["disabled", "rule", "deepseek"], default="rule")
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    model_ref = str(args.reranker_model_path) if args.reranker_model_path else args.reranker_model_name
    result = evaluate_stage5(
        scenario_queries_path=args.scenario_queries,
        short_queries_path=args.short_queries,
        guardrail_queries_path=args.guardrail_queries,
        out_path=args.out,
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        rerank_top_k=args.rerank_top_k,
        mmr_pre_candidates=args.mmr_pre_candidates,
        mmr_top_k=args.mmr_top_k,
        token_budget=args.token_budget,
        device=args.device,
        batch_size=args.batch_size,
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        reranker_model_ref=model_ref,
        fake_reranker=args.fake_reranker,
        fake_mmr=args.fake_mmr,
        fake_hyde=args.fake_hyde,
        hyde_mode=args.hyde_mode,
        mmr_lambda=args.mmr_lambda,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
