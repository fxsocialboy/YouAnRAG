"""CLI demo for Stage6 Agentic RAG answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.agent.service import RagAnswerService, RagAnswerServiceOptions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage6 Agentic RAG answer demo")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--sparse-top-k", type=int, default=30)
    parser.add_argument("--branch-candidate-top-k", type=int, default=10)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=True)
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("--mmr", dest="mmr", action="store_true", default=True)
    parser.add_argument("--no-mmr", dest="mmr", action="store_false")
    parser.add_argument("--fake-reranker", action="store_true")
    parser.add_argument("--hyde-mode", choices=["disabled", "rule", "deepseek"], default="disabled")
    parser.add_argument("--min-evidence-score", type=float, default=0.3)
    parser.add_argument("--json", action="store_true", help="Print the full RagAnswer JSON payload")
    return parser.parse_args(argv)


def build_options(args: argparse.Namespace) -> RagAnswerServiceOptions:
    return RagAnswerServiceOptions(
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        branch_candidate_top_k=args.branch_candidate_top_k,
        token_budget=args.token_budget,
        enable_reranker=args.rerank,
        enable_mmr=args.mmr,
        min_evidence_score=args.min_evidence_score,
    )


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    options = build_options(args)
    service = RagAnswerService.from_config(
        device=args.device,
        batch_size=args.batch_size,
        fake_reranker=args.fake_reranker,
        hyde_mode=args.hyde_mode,
        enable_reranker=args.rerank,
        enable_mmr=args.mmr,
        default_options=options,
    )
    result = service.answer(args.query, options=options)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        if result.fallback_reason:
            print(f"\n[fallback_reason] {result.fallback_reason}")


if __name__ == "__main__":
    main()
