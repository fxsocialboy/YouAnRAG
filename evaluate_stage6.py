"""Lightweight Stage6 engineering evaluation.

The evaluation deliberately uses an in-memory fake Stage5 pipeline.  Stage4/5
already measure retrieval quality with real models; Stage6 closes a different
contract: every successful answer must carry resolvable citations and trace
data, while empty/weak evidence must fail closed.  It therefore runs quickly
on CPU and is deterministic enough for unit tests and CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.agent.legacy_adapter import invoke
from rag_v2.agent.service import RagAnswerService, RagAnswerServiceOptions
from rag_v2.agent.verifier import CitationVerifier
from rag_v2.query.models import QueryBranch, QueryPlan
from rag_v2.retrieval.hybrid_searcher import HybridSearchResult
from rag_v2.retrieval.multi_query_pipeline import MultiQuerySearchOutput


@dataclass(frozen=True)
class EvalCase:
    query: str
    content: str | None
    score: float = 0.9
    expected_fallback: bool = False


CASES = (
    EvalCase("台风黄色预警下学校应该怎么做", "学校应停止露天集体活动，并检查门窗和临时设施。"),
    EvalCase("地震后高层建筑居民如何疏散", "高层建筑人员应听从指挥，沿安全通道有序疏散，不乘坐电梯。"),
    EvalCase("IV级气象灾害应急响应由谁启动", "达到启动标准后，由气象灾害应急指挥机构决定启动 IV 级响应。"),
    EvalCase("暴雨预警期间地下空间要采取什么措施", "地下空间管理单位应加强巡查，准备挡水和排水设施。"),
    EvalCase("山洪来临时群众应该往哪里转移", "群众应迅速向沟谷两侧高地转移，避开河道和低洼区域。"),
    EvalCase("学校发现火灾后如何组织学生撤离", "学校应立即报警，并按疏散预案组织学生从安全出口撤离。"),
    EvalCase("极端高温时户外作业如何安排", "用人单位应调整户外作业时间，避开高温时段并提供防暑物资。"),
    EvalCase("应急物资储备需要记录哪些信息", "应记录物资名称、数量、存放位置、有效期和责任人。"),
    EvalCase("知识库完全没有覆盖的量子通信问题", None, expected_fallback=True),
    EvalCase("一个只有微弱相关证据的问题", "这是一条相关性不足的模糊材料。", score=0.1, expected_fallback=True),
)


class FakeStage5Pipeline:
    def __init__(self, case: EvalCase):
        self.case = case

    def search(self, query: str, *, filters=None, options=None) -> MultiQuerySearchOutput:
        results: list[HybridSearchResult] = []
        if self.case.content is not None:
            results.append(
                HybridSearchResult(
                    chunk_id=f"stage6_eval.md::{hashlib.sha1(query.encode('utf-8')).hexdigest()[:10]}",
                    content=self.case.content,
                    source_file="stage6_eval.md",
                    chunk_index=0,
                    section_path_text="评测 / 应急处置",
                    dense_score=self.case.score,
                    sparse_score=self.case.score,
                    fusion_score=self.case.score,
                    rank=1,
                    metadata={"matched_branches": ["raw"], "token_count": 32},
                    rerank_score=self.case.score,
                    mmr_score=self.case.score,
                    stage4_rank=1,
                )
            )
        return MultiQuerySearchOutput(
            query_plan=QueryPlan(original_query=query, normalized_query=query, query_type="scenario"),
            query_branches=[QueryBranch(branch="raw", query=query, weight=1.0, reason="raw_query")],
            results=results,
            hyde_document=None,
        )


def _markers_resolve(answer_text: str, evidence: list[Any]) -> bool:
    known = {item.marker for item in evidence}
    import re

    used = set(re.findall(r"\[S\d+\]", answer_text))
    return bool(used) and used.issubset(known)


def run_evaluation(output_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        service = RagAnswerService(FakeStage5Pipeline(case))
        answer = service.answer(case.query, options=RagAnswerServiceOptions(min_evidence_score=0.3))
        if answer.is_fallback:
            # A fallback answer intentionally contains no factual citation.  It
            # is valid as long as it does not expose an unresolved marker; low
            # score evidence may remain in trace/evidence for diagnostics.
            import re

            known = {item.marker for item in answer.evidence}
            used = set(re.findall(r"\[S\d+\]", answer.answer))
            citation_valid = used.issubset(known)
        else:
            citation_valid = _markers_resolve(answer.answer, answer.evidence)
        trace_complete = bool(answer.trace.query_plan) and bool(answer.trace.branches) and "evidence_count" in answer.trace.extra
        rows.append(
            {
                "query": case.query,
                "expected_fallback": case.expected_fallback,
                "is_fallback": answer.is_fallback,
                "fallback_reason": answer.fallback_reason,
                "citation_count": len(answer.citations),
                "evidence_count": len(answer.evidence),
                "citation_valid": citation_valid,
                "trace_complete": trace_complete,
                "passed": answer.is_fallback == case.expected_fallback and citation_valid and trace_complete,
            }
        )

    normal_case = CASES[0]
    normal_service = RagAnswerService(FakeStage5Pipeline(normal_case))
    legacy_chunks = invoke(normal_case.query, top_k=1, service=normal_service)
    normal_answer = normal_service.answer(normal_case.query)
    invalid_citation_detected = not CitationVerifier().verify(
        normal_answer.answer + " [S999]", normal_answer.citations, normal_answer.evidence
    ).passed

    summary = {
        "query_count": len(rows),
        "answer_success_count": sum(not row["is_fallback"] for row in rows),
        "expected_fallback_count": sum(row["expected_fallback"] for row in rows),
        "fallback_count": sum(row["is_fallback"] for row in rows),
        "citation_valid_count": sum(row["citation_valid"] for row in rows),
        "trace_complete_count": sum(row["trace_complete"] for row in rows),
        "guardrail_trigger_count": sum(row["is_fallback"] for row in rows),
        "all_citations_mapped": all(row["citation_valid"] for row in rows),
        "invalid_citation_detected": invalid_citation_detected,
        "legacy_adapter_ok": legacy_chunks == [normal_case.content],
        "passed_case_count": sum(row["passed"] for row in rows),
    }
    summary["passed"] = (
        summary["passed_case_count"] == summary["query_count"]
        and summary["all_citations_mapped"]
        and summary["invalid_citation_detected"]
        and summary["legacy_adapter_ok"]
    )
    report = {
        "stage": "6.6",
        "evaluation_scope": "deterministic_agent_contract",
        "uses_gpu": False,
        "uses_external_llm": False,
        "summary": summary,
        "cases": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lightweight Stage6 Agent contract evaluation")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "stage6" / "stage6_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_evaluation(args.out)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if not report["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
