"""Run the original YouAn DisasterResponseAgent with the real RAG V2 backend."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
EXTERNAL_SERVER_ROOT = PROJECT_ROOT.parent / "Youan-AI-main" / "youan-multiagent" / "multi_agent_server"
BUNDLED_SERVER_ROOT = PROJECT_ROOT / "legacy_snapshot" / "multi_agent_server"
OLD_SERVER_ROOT = EXTERNAL_SERVER_ROOT if EXTERNAL_SERVER_ROOT.exists() else BUNDLED_SERVER_ROOT
for path in (SRC, OLD_SERVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.agents.disaster_response_agent import DisasterResponseAgent
from rag_v2.agent.legacy_adapter import RagV2RetrieverAdapter
from rag_v2.config import get_config
from rag_v2.llm.deepseek_client import DeepSeekChatClient


SCENARIOS = [
    "深圳市气象灾害Ⅳ级响应由哪个机构决定启动？请给出简要响应方案。",
    "山区村庄发现滑坡和落石迹象后，应如何组织群众提前避险？",
    "洪涝突发险情发生后，应由谁负责掌握和报告灾情？",
]


class FakeAgentLLM:
    def invoke_messages(self, messages, **kwargs):
        if "评审委员会" in messages[0]["content"]:
            return "方案结构完整，请在最终版本中继续保留全部证据引用。"
        return "已根据参考知识形成应急方案，关键措施和责任主体见引用。[S1]"


class DeepSeekAgentLLM:
    def __init__(self, client: DeepSeekChatClient):
        self.client = client

    def invoke_messages(self, messages, **kwargs):
        return self.client.complete(
            messages,
            temperature=float(kwargs.get("temperature", 0.2)),
            max_tokens=1400,
        )


def main() -> None:
    # Windows PowerShell may still use GBK while the legacy Agent prints
    # Unicode status symbols.  Keep the smoke test portable across Windows
    # and Linux without changing the Agent's business behaviour.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-llm", action="store_true")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--no-mmr", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "stage7" / "youan_agent_real_smoke.json")
    args = parser.parse_args()

    cfg = get_config()
    embedding_path = PROJECT_ROOT / "models" / "bge-large-zh-v1.5"
    reranker_path = PROJECT_ROOT / "models" / "bge-reranker-base"
    cfg = replace(
        cfg,
        model_path=embedding_path if embedding_path.exists() else cfg.model_path,
        reranker_model_path=reranker_path if reranker_path.exists() else cfg.reranker_model_path,
    )
    retriever = RagV2RetrieverAdapter.from_config(
        cfg=cfg,
        top_k=5,
        device=args.device,
        reranker_device=args.reranker_device or args.device,
        enable_reranker=not args.no_reranker,
        enable_mmr=not args.no_mmr,
    )
    if args.real_llm:
        client = DeepSeekChatClient(cfg.deepseek_api_key or "", model=cfg.deepseek_model, timeout=40, max_retries=1)
        proposer, critic = DeepSeekAgentLLM(client), DeepSeekAgentLLM(client)
    else:
        proposer, critic = FakeAgentLLM(), FakeAgentLLM()
    agent = DisasterResponseAgent(
        retriever=retriever,
        proposer_llm=proposer,
        critique_llm=critic,
        max_iterations=1,
    )
    rows = []
    for scenario in SCENARIOS[: max(args.limit, 0)]:
        result = agent.generate_plan_with_trace(scenario, verbose=False)
        rows.append({"scenario": scenario, **result})
    payload = {
        "stage": "7.3",
        "rag_backend": "v2",
        "real_retrieval": True,
        "real_llm": args.real_llm,
        "case_count": len(rows),
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"case_count": len(rows), "real_llm": args.real_llm, "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
