"""Stage7 CLI for template/DeepSeek answers and optional DeepSeek judging."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.agent.service import RagAnswerService, RagAnswerServiceOptions
from rag_v2.config import get_config
from rag_v2.evaluation.deepseek_judge import DeepSeekAnswerJudge
from rag_v2.llm.deepseek_client import DeepSeekChatClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage7 Agentic RAG answer + judge demo")
    parser.add_argument("query")
    parser.add_argument("--composer-mode", choices=["template", "deepseek"], default="deepseek")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--sparse-top-k", type=int, default=30)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--reranker-model-path", type=Path, default=None)
    parser.add_argument("--fake-reranker", action="store_true")
    parser.add_argument("--hyde-mode", choices=["disabled", "rule", "deepseek"], default="disabled")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-mmr", action="store_true")
    return parser.parse_args(argv)


def build_options(args: argparse.Namespace) -> RagAnswerServiceOptions:
    return RagAnswerServiceOptions(
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        token_budget=args.token_budget,
        enable_reranker=not args.no_rerank,
        enable_mmr=not args.no_mmr,
        composer_mode=args.composer_mode,
    )


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    cfg = get_config()
    if args.reranker_model_path is not None:
        cfg = replace(cfg, reranker_model_path=args.reranker_model_path.resolve())
    options = build_options(args)
    service = RagAnswerService.from_config(
        cfg=cfg,
        device=args.device,
        batch_size=args.batch_size,
        reranker_device=args.reranker_device,
        fake_reranker=args.fake_reranker,
        hyde_mode=args.hyde_mode,
        enable_reranker=not args.no_rerank,
        enable_mmr=not args.no_mmr,
        default_options=options,
    )
    answer = service.answer(args.query, options=options)
    payload = {"rag_answer": answer.to_dict(), "judge": None}
    if args.judge and answer.evidence and not answer.is_fallback:
        client = DeepSeekChatClient(
            api_key=cfg.deepseek_api_key or "",
            model=cfg.deepseek_model,
            timeout=max(cfg.deepseek_timeout, 20),
            max_retries=cfg.deepseek_max_retries,
        )
        payload["judge"] = DeepSeekAnswerJudge(client).evaluate(
            query=args.query,
            answer=answer.answer,
            evidence=answer.evidence,
            composer_mode=answer.trace.composer_mode,
        ).to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
