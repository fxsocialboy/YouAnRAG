"""CPU-safe preflight for the Stage 7.4 after-fix AutoDL run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "models/bge-large-zh-v1.5/config.json", "models/bge-reranker-base/config.json",
    "artifacts/stage1/faiss_index.index", "artifacts/stage1/chunk_metadata.json",
    "artifacts/stage3/bm25_index.json", "legacy_snapshot/RAG/faiss_index.index",
    "legacy_snapshot/RAG/chunk_metadata.json", "experiments/eval_queries_final_labeled.jsonl",
    "experiments/eval_queries_final_random.jsonl",
)


def run(*, require_cuda: bool = True, check_deepseek: bool = False) -> dict:
    checks = []
    for relative in REQUIRED:
        path = ROOT / relative
        checks.append({"name": relative, "passed": path.exists(), "detail": str(path)})
    try:
        import torch
        cuda = torch.cuda.is_available()
        checks.append({"name": "cuda", "passed": cuda or not require_cuda, "detail": torch.cuda.get_device_name(0) if cuda else "unavailable"})
    except Exception as exc:
        checks.append({"name": "cuda", "passed": not require_cuda, "detail": str(exc)})
    for relative in ("experiments/eval_queries_final_labeled.jsonl", "experiments/eval_queries_final_random.jsonl"):
        path = ROOT / relative
        if path.exists():
            checks.append({"name": relative + ":sha256", "passed": True, "detail": hashlib.sha256(path.read_bytes()).hexdigest()})
    qdrant_meta = ROOT / "artifacts/qdrant_local/meta.json"
    checks.append({"name": "qdrant_local", "passed": qdrant_meta.exists(), "detail": str(qdrant_meta)})
    if check_deepseek:
        try:
            if str(ROOT / "src") not in sys.path:
                sys.path.insert(0, str(ROOT / "src"))
            from rag_v2.config import get_config
            from rag_v2.llm.deepseek_client import DeepSeekChatClient
            cfg = get_config()
            client = DeepSeekChatClient(
                api_key=cfg.deepseek_api_key or "", model=cfg.deepseek_model,
                timeout=max(cfg.deepseek_timeout, 20), max_retries=0,
            )
            reply = client.complete(
                [{"role": "user", "content": "只回复OK"}], temperature=0.0, max_tokens=8
            )
            checks.append({"name": "deepseek_smoke", "passed": bool(reply), "detail": reply[:40]})
        except Exception as exc:
            checks.append({"name": "deepseek_smoke", "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--skip-deepseek-smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/stage7/after_fix/preflight.json")
    args = parser.parse_args()
    report = run(require_cuda=not args.allow_cpu, check_deepseek=not args.skip_deepseek_smoke)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
