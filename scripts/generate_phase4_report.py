"""Generate Stage4 closing report from rerank evaluation artifacts."""

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

VARIANTS = ["hybrid", "hybrid_rerank", "hybrid_mmr", "hybrid_rerank_mmr"]
VARIANT_LABELS = {
    "hybrid": "hybrid",
    "hybrid_rerank": "hybrid + rerank",
    "hybrid_mmr": "hybrid + MMR",
    "hybrid_rerank_mmr": "hybrid + rerank + MMR",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _metric_table(summary: dict[str, Any]) -> str:
    variants = summary.get("variants", {})
    lines = [
        "| Variant | doc_recall@10 | chunk_recall@10 | doc_MRR@10 | chunk_MRR@10 | doc_MRR_delta | avg_doc_rank_shift | avg_latency_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        m = variants.get(variant, {})
        lines.append(
            f"| {VARIANT_LABELS[variant]} | "
            f"{m.get('doc_recall@10', 'N/A')} | "
            f"{m.get('chunk_recall_exact@10', 'N/A')} | "
            f"{m.get('doc_mrr@10', 'N/A')} | "
            f"{m.get('chunk_mrr_exact@10', 'N/A')} | "
            f"{m.get('doc_mrr_delta_vs_hybrid', 'N/A')} | "
            f"{m.get('avg_doc_rank_shift_vs_hybrid', 'N/A')} | "
            f"{m.get('avg_latency_ms', 'N/A')} |"
        )
    return "\n".join(lines)


def _pick_shift_examples(rows: list[dict[str, Any]], variant: str, limit: int = 3) -> list[dict[str, Any]]:
    improved = []
    for row in rows:
        metrics = row.get("variant_metrics", {}).get(variant, {})
        shift = metrics.get("doc_rank_shift_vs_hybrid")
        if shift is not None and shift < 0:
            improved.append(row)
    improved.sort(key=lambda row: row["variant_metrics"][variant]["doc_rank_shift_vs_hybrid"])
    return improved[:limit]


def generate_phase4_report(project_root: Path = PROJECT_ROOT) -> Path:
    cfg = get_config()
    stage4_dir = cfg.stage4_artifacts_dir
    stage4_dir.mkdir(parents=True, exist_ok=True)
    eval_path = project_root / "experiments" / "stage4_rerank_eval.json"
    evaluation = read_json(eval_path)
    summary = evaluation.get("summary", {})
    natural_summary = summary.get("natural_queries", {})
    keyword_summary = summary.get("keyword_queries", {})
    natural_rows = evaluation.get("natural_results", [])
    keyword_rows = evaluation.get("keyword_results", [])
    examples = _pick_shift_examples(natural_rows + keyword_rows, "hybrid_rerank_mmr")

    report_path = stage4_dir / "phase4_report.md"
    text = f"""# 阶段四收口报告：Cross-Encoder 精排与 MMR 多样性控制

## 1. 阶段四目标

阶段四在阶段三 `Qdrant dense + BM25 sparse + RRF` 混合召回基础上，引入 Cross-Encoder rerank 和 MMR 多样性选择，把粗排候选进一步加工为更适合进入 LLM 的 evidence。

本阶段关注三件事：

1. reranker 是否把相关证据前移；
2. MMR 是否能减少最终候选同质化；
3. Stage4 是否能在异常时退化回 Stage3 hybrid，不破坏已有链路。

## 2. 已完成能力

| 能力 | 状态 |
|---|---|
| `HybridSearchResult` 扩展 `rerank_score/mmr_score/stage4_rank` | 完成 |
| `CrossEncoderReranker` + `FakeReranker` | 完成 |
| `MMRSelector` | 完成 |
| `ContextPacker` 三级降级排序：`mmr_score > rerank_score > fusion_score` | 完成 |
| `RerankPipeline` + Stage4 CLI | 完成 |
| Stage4 四路评测与报告 | 完成 |

## 3. 评测设置

评测文件：

```text
G:\\tiaozhanbei\\newrag\\experiments\\stage4_rerank_eval.json
```

本次评测配置：

```json
{json.dumps({
    'top_k': summary.get('top_k'),
    'dense_top_k': summary.get('dense_top_k'),
    'sparse_top_k': summary.get('sparse_top_k'),
    'rerank_top_k': summary.get('rerank_top_k'),
    'mmr_pre_candidates': summary.get('mmr_pre_candidates'),
    'mmr_top_k': summary.get('mmr_top_k'),
    'fake_reranker': summary.get('fake_reranker'),
    'fake_mmr': summary.get('fake_mmr'),
    'mmr_lambda': summary.get('mmr_lambda'),
    'reranker_model_ref': summary.get('reranker_model_ref'),
}, ensure_ascii=False, indent=2)}
```

对比四路：

```text
hybrid
hybrid + rerank
hybrid + MMR
hybrid + rerank + MMR
```

## 4. 自然语言 query 结果

{_metric_table(natural_summary)}

## 5. 关键词 / 条款 query 结果

{_metric_table(keyword_summary)}

## 6. 结果解读

- `doc_recall@10` / `chunk_recall_exact@10` 说明相关证据是否进入 Top-10；
- `MRR@10` 说明相关证据排在多靠前，是阶段四最核心指标；
- `rank_shift` 为负数表示相关文档或 chunk 被前移；
- `avg_latency_ms` 用于衡量 rerank/MMR 带来的额外代价。

如果 `fake_reranker=true`，本报告只说明链路可跑通，不代表真实 reranker 效果。真实收口建议使用本地轻量真实 smoke 或 AutoDL 完整评测结果覆盖本文件。

## 7. 代表性 rank 前移样例

"""
    if examples:
        for row in examples:
            metrics = row["variant_metrics"]["hybrid_rerank_mmr"]
            text += (
                f"- `{row.get('id')}` / `{row.get('query')}`："
                f"doc_rank_shift={metrics.get('doc_rank_shift_vs_hybrid')}，"
                f"chunk_rank_shift={metrics.get('chunk_rank_shift_vs_hybrid')}\n"
            )
    else:
        text += "- 当前结果中没有出现明显 rank 前移样例；如果是 fake reranker，这是正常现象。\n"

    text += """

## 8. 阶段四产物

```text
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\retrieval\\reranker.py
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\retrieval\\mmr.py
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\retrieval\\rerank_pipeline.py
G:\\tiaozhanbei\\newrag\\scripts\\search_rerank_stage4.py
G:\\tiaozhanbei\\newrag\\evaluate_stage4.py
G:\\tiaozhanbei\\newrag\\scripts\\generate_phase4_report.py
G:\\tiaozhanbei\\newrag\\artifacts\\stage4\\phase4_report.md
```

## 9. GPU / AutoDL 建议

本地 CPU 可以跑：

```powershell
python evaluate_stage4.py --fake-reranker --fake-mmr --limit 2 --device cpu
python scripts\\search_rerank_stage4.py "IV????????????????" --top-k 5 --device cpu --reranker-model-path G:\\tiaozhanbei\\newrag\\models\\bge-reranker-base --no-mmr
```

完整真实评测建议在 GPU / AutoDL 上跑：

```bash
python evaluate_stage4.py --device cuda --reranker-device cuda --reranker-model-path models/bge-reranker-base
```

## 10. 阶段四简历表述

> 在 dense+sparse 混合召回的 RRF 粗排基础上，引入 Cross-Encoder reranker 对 query-passage pair 做联合建模精排，并使用 MMR 在相关性与多样性之间做折中；设计 `mmr_score > rerank_score > fusion_score` 的三级降级排序，使精排和多样性结果真正进入最终 evidence pack；通过 MRR@10、rank_shift、recall 和 latency 对 hybrid/rerank/MMR 进行消融评测，同时保留开关和异常 fallback，保证可回退到阶段三 hybrid 链路。
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    path = generate_phase4_report()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
