"""Generate Stage5 query planning / HyDE evaluation report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.config import get_config

VARIANTS = ["stage4_raw", "stage5_rewrite", "stage5_multi_query", "stage5_hyde"]
LABELS = {
    "stage4_raw": "Stage4 raw",
    "stage5_rewrite": "Stage5 rewrite",
    "stage5_multi_query": "Stage5 multi-query",
    "stage5_hyde": "Stage5 HyDE",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric_table(group: dict[str, Any], *, guardrail: bool = False) -> str:
    lines = [
        "| Variant | doc_recall@10 | chunk_recall@10 | doc_MRR@10 | chunk_MRR@10 | doc_MRR_delta | avg_branch_count | hyde_used_ratio | avg_latency_ms |" + (" guardrail |" if guardrail else " |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if guardrail else ""),
    ]
    variants = group.get("variants", {})
    for variant in VARIANTS:
        m = variants.get(variant, {})
        row = (
            f"| {LABELS[variant]} | {m.get('doc_recall@10', 'N/A')} | {m.get('chunk_recall_exact@10', 'N/A')} | "
            f"{m.get('doc_mrr@10', 'N/A')} | {m.get('chunk_mrr_exact@10', 'N/A')} | "
            f"{m.get('doc_mrr_delta_vs_stage4', 'N/A')} | {m.get('avg_branch_count', 'N/A')} | "
            f"{m.get('hyde_used_ratio', 'N/A')} | {m.get('avg_latency_ms', 'N/A')} |"
        )
        if guardrail:
            status = m.get("guardrail_no_regression")
            marker = "✅" if status is True or variant == "stage4_raw" else "🔴"
            row += f" {marker} |"
        lines.append(row)
    return "\n".join(lines)


def generate_phase5_report(project_root: Path = PROJECT_ROOT) -> Path:
    cfg = get_config()
    stage5_dir = cfg.artifacts_dir / "stage5"
    stage5_dir.mkdir(parents=True, exist_ok=True)
    eval_path = project_root / "experiments" / "stage5_query_eval.json"
    evaluation = read_json(eval_path)
    summary = evaluation.get("summary", {})
    report_path = stage5_dir / "phase5_report.md"
    guardrail_summary = summary.get("exact_guardrail_queries", {})
    guardrail_failed = [
        variant for variant, metrics in guardrail_summary.get("variants", {}).items()
        if variant != "stage4_raw" and metrics.get("guardrail_no_regression") is False
    ]
    guardrail_text = "通过" if not guardrail_failed else "未通过：" + ", ".join(guardrail_failed)
    text = f"""# 阶段五收口报告：Query Analyzer / Rewrite / Multi-Query / HyDE

## 1. 阶段五目标

阶段五在阶段四 `dense + sparse + rerank + MMR` 链路前新增 Query Planning 层：先识别 query 类型，再决定是否启用规则改写、Multi-Query 和 HyDE。核心目标不是盲目堆叠 HyDE，而是让不同 query 类型走更合适的检索策略。

## 2. 评测配置

```json
{json.dumps({k: summary.get(k) for k in ['top_k','dense_top_k','sparse_top_k','rerank_top_k','mmr_pre_candidates','mmr_top_k','fake_reranker','fake_mmr','fake_hyde','hyde_mode','reranker_model_ref']}, ensure_ascii=False, indent=2)}
```

对比四路：

```text
stage4_raw
stage5_rewrite
stage5_multi_query
stage5_hyde
```

## 3. 场景处置类 query

{metric_table(summary.get('scenario_queries', {}))}

## 4. 短 query / 口语 query

{metric_table(summary.get('short_ambiguous_queries', {}))}

## 5. 精确条款守护 query

{metric_table(guardrail_summary, guardrail=True)}

守护结论：**{guardrail_text}**。

## 6. 结果解读

- `stage4_raw` 是阶段四单 query 基线；
- `stage5_rewrite` 用较少 branch 验证规则 rewrite 的收益；
- `stage5_multi_query` 使用 raw / normalized / expanded 多分支；
- `stage5_hyde` 在 multi-query 基础上增加 HyDE 分支；
- `guardrail_no_regression` 用于保证精确条款 query 的 `doc_recall@10` 和 `chunk_recall_exact@10` 不比 Stage4 下降。

## 7. 阶段五产物

```text
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\query\\models.py
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\query\\analyzer.py
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\query\\rewriter.py
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\query\\hyde.py
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\retrieval\\multi_query_pipeline.py
G:\\tiaozhanbei\\newrag\\scripts\\search_stage5.py
G:\\tiaozhanbei\\newrag\\evaluate_stage5.py
G:\\tiaozhanbei\\newrag\\scripts\\generate_phase5_report.py
```

## 8. 简历表述

> 在混合召回与 Cross-Encoder 精排基础上，我进一步实现了 Query Planning 层：通过规则 Query Analyzer 识别场景类、短 query、精确条款等不同 query 类型，并生成 raw / normalized / expanded / HyDE 多分支检索请求；使用 branch-aware RRF 融合多分支结果，保留 raw query 最高权重，并将 HyDE 限制为 dense-only 分支，禁止进入 evidence/citation。评测上加入精确条款 guardrail，确保 query rewrite 和 HyDE 不破坏原本有效的关键词检索。
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    path = generate_phase5_report()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
