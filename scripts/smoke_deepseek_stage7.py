"""Real DeepSeek Composer + Judge smoke without loading embedding models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.agent.composer import TemplateAnswerComposer
from rag_v2.agent.llm_composer import DeepSeekAnswerComposer
from rag_v2.agent.models import EvidenceItem
from rag_v2.config import get_config
from rag_v2.evaluation.deepseek_judge import DeepSeekAnswerJudge
from rag_v2.llm.deepseek_client import DeepSeekChatClient


CASES = [
    {
        "query": "Ⅳ级气象灾害应急响应命令由谁签发？",
        "source_file": "国家气象灾害应急预案-2.md",
        "chunk_id": "国家气象灾害应急预案-2.md::13",
        "section": "应急响应命令",
        "content": "原则上，Ⅳ级和Ⅲ级应急响应命令由副局长签发，Ⅱ级和Ⅰ级应急响应命令由局长签发。",
    },
    {
        "query": "深圳市气象灾害Ⅳ级响应由哪个机构决定启动？",
        "source_file": "深圳市气象灾害应急预案.md",
        "chunk_id": "深圳市气象灾害应急预案.md::48",
        "section": "Ⅳ级应急响应",
        "content": "由市气象灾害指挥部决定是否启动Ⅳ级应急响应，并授权市减灾委办公室和市气象灾害指挥部办公室联合签发。",
    },
    {
        "query": "洪涝突发险情灾情由谁负责掌握和报告？",
        "source_file": "洪涝突发险情灾情报告暂行规定.md",
        "chunk_id": "洪涝突发险情灾情报告暂行规定.md::1",
        "section": "报告责任",
        "content": "各级防汛抗旱指挥部负责本地区洪涝突发险情灾情的及时掌握与报告工作，并确定专人负责。",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "stage7" / "deepseek_stage72_smoke.json")
    args = parser.parse_args()
    cfg = get_config()
    client = DeepSeekChatClient(
        api_key=cfg.deepseek_api_key or "",
        model=cfg.deepseek_model,
        timeout=30,
        max_retries=cfg.deepseek_max_retries,
    )
    composer = DeepSeekAnswerComposer(client, TemplateAnswerComposer())
    judge = DeepSeekAnswerJudge(client)
    rows = []
    for case in CASES:
        evidence = [
            EvidenceItem(
                "S1",
                case["chunk_id"],
                case["source_file"],
                [case["section"]],
                case["content"],
                0.95,
                1,
            )
        ]
        result = composer.compose_with_trace(case["query"], evidence)
        judged = judge.evaluate(
            query=case["query"],
            answer=result.answer,
            evidence=evidence,
            composer_mode=result.actual_mode,
        )
        rows.append(
            {
                "query": case["query"],
                "answer": result.answer,
                "composer": result.to_dict(),
                "judge": judged.to_dict(),
            }
        )
    summary = {
        "case_count": len(rows),
        "deepseek_success_count": sum(row["composer"]["actual_mode"] == "deepseek" for row in rows),
        "avg_faithfulness": sum(row["judge"]["faithfulness"] for row in rows) / len(rows),
        "avg_answer_relevancy": sum(row["judge"]["answer_relevancy"] for row in rows) / len(rows),
        "avg_citation_correctness": sum(row["judge"]["citation_correctness"] or 0.0 for row in rows) / len(rows),
        "avg_citation_completeness": sum(row["judge"]["citation_completeness"] for row in rows) / len(rows),
    }
    payload = {"stage": "7.2", "summary": summary, "cases": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
