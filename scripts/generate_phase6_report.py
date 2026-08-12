"""Generate the Stage6 closure report from the deterministic evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_phase6_report(eval_path: Path, report_path: Path) -> Path:
    data: dict[str, Any] = json.loads(eval_path.read_text(encoding="utf-8"))
    summary = data["summary"]
    rows = []
    for index, case in enumerate(data["cases"], 1):
        rows.append(
            f"| {index} | {case['query']} | {case['is_fallback']} | {case['citation_valid']} | "
            f"{case['trace_complete']} | {'通过' if case['passed'] else '失败'} |"
        )
    status = "通过" if summary["passed"] else "未通过"
    text = f"""# 阶段六收口报告：Agent 服务化、可信引用与答案质量

## 1. 收口结论

阶段六工程闭环评测：**{status}**。项目已经从检索组件升级为可通过 Python API、CLI 和 FastAPI 调用的 Agentic RAG 问答服务。

## 2. 评测边界

- 本评测验证 Agent 层的数据合同、引用映射、trace、证据不足降级和 legacy adapter；
- 采用确定性的 fake Stage5 pipeline，不使用 GPU 和外部 LLM；
- 真实召回、rerank 与 Query Planning 的效果已分别由阶段三至阶段五评测覆盖，阶段六不重复进行大规模检索测评。

## 3. 核心指标

| 指标 | 结果 |
|---|---:|
| query_count | {summary['query_count']} |
| answer_success_count | {summary['answer_success_count']} |
| fallback_count | {summary['fallback_count']} |
| citation_valid_count | {summary['citation_valid_count']} |
| trace_complete_count | {summary['trace_complete_count']} |
| invalid_citation_detected | {summary['invalid_citation_detected']} |
| legacy_adapter_ok | {summary['legacy_adapter_ok']} |
| passed_case_count | {summary['passed_case_count']} |

## 4. 用例结果

| # | Query | Fallback | Citation valid | Trace complete | 结果 |
|---:|---|---:|---:|---:|---|
{chr(10).join(rows)}

## 5. 阶段六产物

- `src/rag_v2/agent/models.py`：Evidence、Citation、Trace、RagAnswer 数据合同；
- `src/rag_v2/agent/service.py`：封装 Stage5 的统一问答服务；
- `src/rag_v2/agent/evidence.py`、`composer.py`：证据编号与可引用模板回答；
- `src/rag_v2/agent/verifier.py`、`guardrail.py`：引用校验与证据不足降级；
- `src/rag_v2/agent/legacy_adapter.py`、`api.py`：旧接口适配与 FastAPI；
- `scripts/answer_stage6.py`：CLI 演示；
- `evaluate_stage6.py`：阶段六确定性工程评测。

## 6. 简历/面试表述

> 将前五阶段的多路检索链路封装为 Agentic RAG 问答服务，设计 Evidence/Citation/RagAnswer 数据合同，实现证据编号、可追溯引用、确定性引用校验与证据不足降级，并通过 legacy adapter、CLI 和 FastAPI 提供兼容旧 Agent 的服务化接口；使用 trace 暴露 Query Plan、检索分支和证据数量，完成可解释、可降级的端到端闭环。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=PROJECT_ROOT / "artifacts" / "stage6" / "stage6_eval.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "stage6" / "phase6_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"wrote {generate_phase6_report(args.eval, args.out)}")


if __name__ == "__main__":
    main()
