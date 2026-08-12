"""Final Stage7 evaluator for AutoDL.

The evaluator deliberately runs one dataset/backend pair per process.  This
keeps GPU memory predictable and makes the three expensive jobs independently
resumable:

1. labeled + legacy: BGE/FAISS retrieval baseline;
2. labeled + v2: complete retrieval, DeepSeek answer and Judge;
3. random + v2: robustness, fallback and answer-quality evaluation.

Every finished row is atomically written to disk.  Re-running the same command
skips successful rows and retries failed/partial rows when ``--retry-errors``
is supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.agent.models import EvidenceItem, RagAnswer
from rag_v2.agent.service import RagAnswerService, RagAnswerServiceOptions
from rag_v2.config import get_config
from rag_v2.evaluation.answer_metrics import citation_mapping_metrics, summarize_answer_rows
from rag_v2.evaluation.deepseek_judge import DeepSeekAnswerJudge
from rag_v2.evaluation.models import EvaluationQuery, load_evaluation_queries, load_stage74_regression_queries
from rag_v2.evaluation.retrieval_metrics import evaluate_ranked_results, summarize_retrieval_rows
from rag_v2.llm.deepseek_client import DeepSeekChatClient


class LegacySearchBackend(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]: ...


class LegacyFaissBackend:
    """Original BGE CLS pooling + normalized FAISS baseline."""

    def __init__(self, *, model_path: Path, index_path: Path, metadata_path: Path, device: str):
        import faiss
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.float32, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.index = faiss.read_index(str(index_path))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.metadata):
            raise ValueError(f"legacy index.ntotal={self.index.ntotal} != metadata={len(self.metadata)}")

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs)
            vector = self.torch.nn.functional.normalize(output.last_hidden_state[:, 0], p=2, dim=1)
        distances, indices = self.index.search(vector.float().cpu().numpy(), top_k)
        rows: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0].tolist(), distances[0].tolist()), 1):
            if idx < 0:
                continue
            item = self.metadata[idx]
            source_file = str(item.get("source_file", ""))
            chunk_index = int(item.get("chunk_index", idx))
            rows.append(
                {
                    "rank": rank,
                    "global_index": idx,
                    "score": float(score),
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "chunk_id": str(item.get("chunk_id") or f"{source_file}::{chunk_index}"),
                    "content_preview": str(item.get("content", ""))[:240],
                }
            )
        return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable final Stage7 evaluation")
    parser.add_argument("--dataset", choices=["labeled", "random", "regression"], required=True)
    parser.add_argument("--backend", choices=["legacy", "v2"], required=True)
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-top-k", type=int, default=40)
    parser.add_argument("--sparse-top-k", type=int, default=40)
    parser.add_argument("--rerank-top-k", type=int, default=30)
    parser.add_argument("--token-budget", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--embedding-model-path", type=Path, default=PROJECT_ROOT / "models" / "bge-large-zh-v1.5")
    parser.add_argument("--reranker-model-path", type=Path, default=PROJECT_ROOT / "models" / "bge-reranker-base")
    parser.add_argument("--legacy-index-path", type=Path, default=PROJECT_ROOT / "legacy_snapshot" / "RAG" / "faiss_index.index")
    parser.add_argument("--legacy-metadata-path", type=Path, default=PROJECT_ROOT / "legacy_snapshot" / "RAG" / "chunk_metadata.json")
    parser.add_argument("--composer-mode", choices=["template", "deepseek"], default="deepseek")
    parser.add_argument("--hyde-mode", choices=["disabled", "rule", "deepseek"], default="deepseek")
    parser.add_argument("--min-evidence-score", type=float, default=0.3)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing output file")
    return parser.parse_args(argv)


def _default_queries(dataset: str) -> Path:
    names = {
        "labeled": "eval_queries_final_labeled.jsonl",
        "random": "eval_queries_final_random.jsonl",
        "regression": "stage74_fix_regression.jsonl",
    }
    name = names[dataset]
    return PROJECT_ROOT / "experiments" / name


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _environment() -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    try:
        import torch

        data.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        data["torch_error"] = str(exc)
    return data


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_existing(
    path: Path, *, dataset: str, backend: str, fresh: bool, config_hash: str | None = None
) -> dict[str, Any] | None:
    if fresh or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset or payload.get("backend") != backend:
        raise ValueError("existing output belongs to a different dataset/backend; use another --out or --fresh")
    if config_hash is not None and payload.get("config_hash") != config_hash:
        raise ValueError("existing output config hash differs; use --fresh or another --out")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _results_from_answer(answer: RagAnswer) -> list[dict[str, Any]]:
    """Expand packed evidence back to rankable source/chunk ids."""

    rows: list[dict[str, Any]] = []
    for evidence in answer.evidence:
        chunk_ids = list(evidence.metadata.get("chunk_ids") or [evidence.chunk_id])
        for chunk_id in chunk_ids:
            rows.append(
                {
                    "rank": evidence.rank,
                    "source_file": evidence.source_file,
                    "chunk_id": str(chunk_id),
                    "score": evidence.score,
                }
            )
    return rows


def _effective_expected_fallback(query: EvaluationQuery) -> bool | None:
    regression_decision = query.metadata.get("stage74_expected_decision")
    if regression_decision in {"answered", "fallback"}:
        return regression_decision == "fallback"
    if query.expected_fallback is not None:
        return query.expected_fallback
    if query.query_type == "out_of_domain" or query.disaster_type == "out_of_domain":
        return True
    return None


def _base_row(query: EvaluationQuery) -> dict[str, Any]:
    row = query.to_dict()
    row["expected_fallback"] = _effective_expected_fallback(query)
    return row


def evaluate_legacy_row(query: EvaluationQuery, backend: LegacySearchBackend, *, top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    results = backend.search(query.query, top_k=top_k)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics = evaluate_ranked_results(
        results,
        relevant_source_files=query.relevant_source_files,
        relevant_chunk_ids=query.relevant_chunk_ids,
        ks=(5, 10),
    )
    metrics["latency_ms"] = latency_ms
    return {
        **_base_row(query),
        "status": "ok",
        "backend": "legacy",
        "retrieval_metrics": metrics,
        "retrieval_results": results,
        "latency_ms": latency_ms,
        "error": None,
    }


def evaluate_v2_row(
    query: EvaluationQuery,
    service: RagAnswerService,
    judge: DeepSeekAnswerJudge | None,
    *,
    options: RagAnswerServiceOptions,
) -> dict[str, Any]:
    started = time.perf_counter()
    answer = service.answer(query.query, options=options)
    answer_latency_ms = round((time.perf_counter() - started) * 1000, 2)
    packed_ranked = _results_from_answer(answer)
    ranked = list(answer.trace.extra.get("raw_retrieval_results") or packed_ranked)
    retrieval_metrics = evaluate_ranked_results(
        ranked,
        relevant_source_files=query.relevant_source_files,
        relevant_chunk_ids=query.relevant_chunk_ids,
        ks=(5, 10),
    )
    retrieval_metrics["latency_ms"] = round(answer.trace.retrieval_latency_ms, 2)
    mapping = citation_mapping_metrics(answer.answer, [item.citation_id for item in answer.citations])
    judge_payload = None
    judge_error = None
    judge_latency_ms = 0.0
    if judge is not None and answer.evidence and not answer.is_fallback:
        judge_started = time.perf_counter()
        try:
            judge_payload = judge.evaluate(
                query=query.query,
                answer=answer.answer,
                evidence=answer.evidence,
                composer_mode=answer.trace.composer_mode,
            ).to_dict()
        except Exception as exc:
            judge_error = {"type": type(exc).__name__, "message": str(exc)}
        judge_latency_ms = round((time.perf_counter() - judge_started) * 1000, 2)

    actual_composer = str(answer.trace.extra.get("composer", {}).get("actual_mode", answer.trace.composer_mode))
    service_error = answer.fallback_reason == "retrieval_exception"
    composer_partial = options.composer_mode == "deepseek" and actual_composer != "deepseek" and not answer.is_fallback
    status = "error" if service_error else ("partial" if judge_error or composer_partial else "ok")
    return {
        **_base_row(query),
        "status": status,
        "backend": "v2",
        "composer_mode": actual_composer,
        "requested_composer_mode": options.composer_mode,
        "is_fallback": answer.is_fallback,
        "fallback_reason": answer.fallback_reason,
        "answer": answer.answer,
        "citations": [item.to_dict() for item in answer.citations],
        "evidence": [item.to_dict() for item in answer.evidence],
        "packed_evidence_results": packed_ranked,
        "trace": answer.trace.to_dict(),
        "retrieval_metrics": retrieval_metrics,
        "retrieval_results": ranked,
        **mapping,
        "judge": judge_payload,
        "judge_error": judge_error,
        "partial_reason": (
            "judge_error" if judge_error else ("deepseek_composer_fell_back" if composer_partial else None)
        ),
        "faithfulness": judge_payload.get("faithfulness") if judge_payload else None,
        "answer_relevancy": judge_payload.get("answer_relevancy") if judge_payload else None,
        "citation_correctness": judge_payload.get("citation_correctness") if judge_payload else None,
        "citation_completeness": judge_payload.get("citation_completeness") if judge_payload else None,
        "answer_latency_ms": answer_latency_ms,
        "online_answer_latency_ms": answer_latency_ms,
        "composer_latency_ms": round(float(answer.trace.extra.get("composer", {}).get("latency_ms", 0.0)), 2),
        "judge_latency_ms": judge_latency_ms,
        "evaluation_total_latency_ms": round(answer_latency_ms + judge_latency_ms, 2),
        "latency_ms": round(answer_latency_ms + judge_latency_ms, 2),
        "error": ({"type": "retrieval_exception", "message": answer.trace.extra.get("exception_message", "")}
                  if service_error else None),
    }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("results", []))
    completed = [row for row in rows if row.get("status") == "ok"]
    summary: dict[str, Any] = {
        "query_count": int(payload.get("query_count", len(rows))),
        "row_count": len(rows),
        "ok_count": len(completed),
        "partial_count": sum(row.get("status") == "partial" for row in rows),
        "error_count": sum(row.get("status") == "error" for row in rows),
        "run_success_rate": round(len(completed) / len(rows), 4) if rows else 0.0,
    }
    retrieval_rows = [row["retrieval_metrics"] for row in rows if row.get("retrieval_metrics")]
    if retrieval_rows:
        summary["retrieval"] = summarize_retrieval_rows(retrieval_rows)
    if payload.get("backend") == "v2":
        summary["answer"] = summarize_answer_rows(rows)
        summary["answer_rate"] = round(sum(not row.get("is_fallback") for row in rows) / len(rows), 4) if rows else 0.0
        summary["fallback_rate"] = round(sum(bool(row.get("is_fallback")) for row in rows) / len(rows), 4) if rows else 0.0
        summary["unknown_citation_count"] = sum(len(row.get("unknown_citation_ids", [])) for row in rows)
        summary["actual_deepseek_count"] = sum(row.get("composer_mode") == "deepseek" for row in rows)
    return summary


def _make_v2(args: argparse.Namespace) -> tuple[RagAnswerService, DeepSeekAnswerJudge | None, RagAnswerServiceOptions]:
    cfg = get_config()
    cfg = replace(
        cfg,
        model_path=args.embedding_model_path.resolve(),
        reranker_model_path=args.reranker_model_path.resolve(),
        reranker_device=args.reranker_device,
        enable_reranker=True,
        enable_mmr=True,
    )
    options = RagAnswerServiceOptions(
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        rerank_top_k=args.rerank_top_k,
        token_budget=args.token_budget,
        enable_reranker=True,
        enable_mmr=True,
        min_evidence_score=args.min_evidence_score,
        composer_mode=args.composer_mode,
    )
    service = RagAnswerService.from_config(
        cfg=cfg,
        device=args.device,
        batch_size=args.batch_size,
        reranker_device=args.reranker_device,
        hyde_mode=args.hyde_mode,
        enable_reranker=True,
        enable_mmr=True,
        default_options=options,
    )
    judge = None
    if not args.no_judge:
        client = DeepSeekChatClient(
            api_key=cfg.deepseek_api_key or "",
            model=cfg.deepseek_model,
            timeout=max(cfg.deepseek_timeout, 40),
            max_retries=max(cfg.deepseek_max_retries, 2),
        )
        judge = DeepSeekAnswerJudge(client)
    return service, judge, options


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset in {"random", "regression"} and args.backend == "legacy":
        raise ValueError("random/regression sets are evaluated only with backend=v2")
    query_path = (args.queries or _default_queries(args.dataset)).resolve()
    if args.dataset == "regression":
        regression_rows = load_stage74_regression_queries(query_path)
        queries = []
        for regression in regression_rows:
            query = regression.to_evaluation_query()
            query.metadata.update(
                {
                    "stage74_expected_decision": regression.expected_decision,
                    "stage74_expected_reason_category": regression.expected_reason_category,
                    "stage74_regression_category": regression.regression_category,
                    "stage74_source_dataset": regression.source_dataset,
                    "stage74_source_query_id": regression.source_query_id,
                }
            )
            queries.append(query)
    else:
        queries = load_evaluation_queries(query_path)
    if args.limit > 0:
        queries = queries[: args.limit]

    config = {
            "top_k": args.top_k,
            "dense_top_k": args.dense_top_k,
            "sparse_top_k": args.sparse_top_k,
            "rerank_top_k": args.rerank_top_k,
            "token_budget": args.token_budget,
            "batch_size": args.batch_size,
            "device": args.device,
            "reranker_device": args.reranker_device,
            "embedding_model_path": str(args.embedding_model_path),
            "reranker_model_path": str(args.reranker_model_path),
            "composer_mode": args.composer_mode,
            "hyde_mode": args.hyde_mode,
            "min_evidence_score": args.min_evidence_score,
            "judge_enabled": not args.no_judge,
            "query_sha256": _sha256(query_path),
    }
    config_hash = _stable_hash(config)
    existing = _load_existing(
        args.out, dataset=args.dataset, backend=args.backend, fresh=args.fresh, config_hash=config_hash
    )
    rows_by_id = {row["query_id"]: row for row in (existing or {}).get("results", [])}
    payload: dict[str, Any] = {
        "stage": "7.4",
        "dataset": args.dataset,
        "backend": args.backend,
        "query_path": str(query_path),
        "query_count": len(queries),
        "config": config,
        "config_hash": config_hash,
        "environment": _environment(),
        "results": [],
    }

    if args.backend == "legacy":
        backend: Any = LegacyFaissBackend(
            model_path=args.embedding_model_path.resolve(),
            index_path=args.legacy_index_path.resolve(),
            metadata_path=args.legacy_metadata_path.resolve(),
            device=args.device,
        )
        service = judge = options = None
    else:
        service, judge, options = _make_v2(args)
        backend = None

    for position, query in enumerate(queries, 1):
        previous = rows_by_id.get(query.query_id)
        if previous and previous.get("status") == "ok":
            print(f"[{position}/{len(queries)}] {query.query_id} skip(ok)", flush=True)
            continue
        if previous and not args.retry_errors:
            print(f"[{position}/{len(queries)}] {query.query_id} skip({previous.get('status')})", flush=True)
            continue
        started = time.perf_counter()
        try:
            if args.backend == "legacy":
                row = evaluate_legacy_row(query, backend, top_k=args.top_k)
            else:
                row = evaluate_v2_row(query, service, judge, options=options)
        except Exception as exc:
            row = {
                **_base_row(query),
                "status": "error",
                "backend": args.backend,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        rows_by_id[query.query_id] = row
        payload["results"] = [rows_by_id[q.query_id] for q in queries if q.query_id in rows_by_id]
        payload["summary"] = summarize(payload)
        _atomic_write(args.out, payload)
        print(
            f"[{position}/{len(queries)}] {query.query_id} status={row['status']} "
            f"latency_ms={row.get('latency_ms')}",
            flush=True,
        )

    payload["results"] = [rows_by_id[q.query_id] for q in queries if q.query_id in rows_by_id]
    payload["summary"] = summarize(payload)
    payload["finished_at_epoch"] = time.time()
    _atomic_write(args.out, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
