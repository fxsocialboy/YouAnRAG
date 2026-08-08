"""Generate Stage3 closing report from hybrid evaluation artifacts."""

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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _delta_block(summary: dict[str, Any]) -> dict[str, float]:
    doc = summary.get("doc_recall@10", {})
    chunk = summary.get("chunk_recall_exact@10", {})
    return {
        "doc_dense": round(float(doc.get("dense", 0.0)), 4),
        "doc_sparse": round(float(doc.get("sparse", 0.0)), 4),
        "doc_hybrid": round(float(doc.get("hybrid", 0.0)), 4),
        "chunk_dense": round(float(chunk.get("dense", 0.0)), 4),
        "chunk_sparse": round(float(chunk.get("sparse", 0.0)), 4),
        "chunk_hybrid": round(float(chunk.get("hybrid", 0.0)), 4),
        "hybrid_minus_dense_doc": round(float(summary.get("hybrid_minus_dense_doc_recall@10", 0.0)), 4),
        "filter_pass_ratio": round(float(summary.get("filter_pass_ratio", 0.0)), 4),
        "packed_nonempty_ratio": round(float(summary.get("packed_nonempty_ratio", 0.0)), 4),
        "avg_packed_token_ratio": round(float(summary.get("avg_packed_token_ratio", 0.0)), 4),
        "avg_latency_ms": round(float(summary.get("avg_latency_ms", 0.0)), 2),
    }


def _pick_examples(rows: list[dict[str, Any]], *, prefer_hybrid_better: bool) -> list[dict[str, Any]]:
    def score(row: dict[str, Any]) -> tuple[int, int, str]:
        hybrid = int(bool(row.get("hybrid_doc_hit@10")))
        dense = int(bool(row.get("dense_doc_hit@10")))
        if prefer_hybrid_better:
            return (hybrid - dense, hybrid, str(row.get("id", "")))
        return (dense - hybrid, dense, str(row.get("id", "")))

    ranked = sorted(rows, key=score, reverse=True)
    picked: list[dict[str, Any]] = []
    for row in ranked:
        if prefer_hybrid_better and row.get("hybrid_doc_hit@10") and not row.get("dense_doc_hit@10"):
            picked.append(row)
        elif (not prefer_hybrid_better) and row.get("dense_doc_hit@10") and not row.get("hybrid_doc_hit@10"):
            picked.append(row)
        if len(picked) >= 3:
            break
    return picked


def generate_phase3_report(project_root: Path = PROJECT_ROOT) -> Path:
    cfg = get_config()
    stage3_dir = cfg.artifacts_dir / "stage3"
    stage3_dir.mkdir(parents=True, exist_ok=True)
    eval_path = project_root / "experiments" / "stage3_hybrid_eval.json"
    evaluation = read_json(eval_path)
    summary = evaluation.get("summary", {})
    natural_summary = summary.get("natural_queries", {})
    keyword_summary = summary.get("keyword_queries", {})
    natural_rows = evaluation.get("natural_results", [])
    keyword_rows = evaluation.get("keyword_results", [])
    natural_block = _delta_block(natural_summary)
    keyword_block = _delta_block(keyword_summary)
    keyword_examples = _pick_examples(keyword_rows, prefer_hybrid_better=True)

    report_path = stage3_dir / "phase3_report.md"
    text = f"""# 阶段三收口报告：混合召回与上下文组装

## 1. 阶段三目标

阶段三的目标是把阶段二已经具备的 **Qdrant 向量检索**，升级为更完整的 **dense + sparse + context packing** 检索链路。

这一阶段重点解决三个问题：

1. 仅靠 dense 检索，对关键词 / 条款 / 数字类问题不稳定；
2. 检索结果只是 Top-K chunk 列表，不能直接作为后续 Rerank / Rewrite / Agent 的 evidence；
3. 需要在不影响阶段一、二稳定运行的前提下，引入可解释的混合召回。

## 2. 已完成的小阶段

| 小阶段 | 内容 | 状态 |
|---|---|---|
| 3.1 | 本地 BM25 稀疏索引与检索 | 完成 |
| 3.2 | Dense + Sparse 的 RRF 融合 | 完成 |
| 3.3 | metadata filter（当前启用 source_file） | 完成 |
| 3.4 | context pack / token budget / section 内补全 | 完成 |
| 3.5 | 评测、报告与阶段三冻结 | 完成 |

## 3. 本阶段新增能力

### 3.1 Sparse 检索

- 基于 Stage1 chunk metadata 构建本地 BM25 索引；
- 不引入 Elasticsearch，保持项目轻量；
- 支持 `source_file` 过滤。

### 3.2 混合召回

- 基于 Qdrant dense 检索与 BM25 sparse 检索；
- 使用 **RRF（Reciprocal Rank Fusion）** 融合候选；
- 保留 `dense_score / sparse_score / fusion_score`，结果更可解释。

### 3.3 软硬过滤

- 先做 query 中的 `source_file` 识别；
- 预留 `region / hazard / doc_type` 识别钩子；
- 不引入 LLM 解析，保证当前阶段收敛。

### 3.4 上下文组装

- 对 Hybrid Top-K 候选做同 section 内邻接扩展；
- 合并相邻 chunk；
- 控制 token budget；
- 输出 citation-ready evidence pack，供后续阶段四 / 五继续使用。

## 4. 评测设置

自然语言 query：25 条  
关键词/条款 query：10 条

评测对比三路：

```text
dense-only
BM25-only
hybrid (dense + sparse + RRF)
```

主结果文件：

```text
G:\\tiaozhanbei\\newrag\\experiments\\stage3_hybrid_eval.json
```

## 5. 评测摘要

### 5.1 自然语言 query（25 条）

```json
{json.dumps(natural_block, ensure_ascii=False, indent=2)}
```

### 5.2 关键词 / 条款 query（10 条）

```json
{json.dumps(keyword_block, ensure_ascii=False, indent=2)}
```

## 6. 结果解读

### 6.1 为什么要做 hybrid

如果自然语言 query 的提升有限，但关键词 / 条款 query 的提升明显，这是符合预期的：

- dense 检索更擅长语义召回；
- sparse 检索更擅长精确关键词、条款、年份、编号；
- hybrid 的价值不在于全面替代 dense，而在于补齐 dense 的短板。

### 6.2 Context packing 的价值

本阶段不仅做了召回，还把结果从：

```text
Top-K chunks
```

升级为了：

```text
citation-ready evidence pack
```

这意味着后续阶段四的 rerank，或者阶段五的 rewrite / multi-query / HyDE，都不需要重新设计 evidence 结构。

## 7. 关键词 query 代表性样例

"""
    if keyword_examples:
        text += "\n"
        for row in keyword_examples:
            top_files = [item.get("source_file", "") for item in row.get("hybrid_top10", [])[:3]]
            text += (
                f"- `{row.get('id')}` / `{row.get('query')}`：dense={int(bool(row.get('dense_doc_hit@10')))}，"
                f"hybrid={int(bool(row.get('hybrid_doc_hit@10')))}，Top-3={top_files}\n"
            )
    else:
        text += "\n- 当前没有出现 `hybrid 命中而 dense 未命中` 的关键词样例，可直接查看完整评测文件。\n"

    text += f"""

## 8. 阶段三产物

```text
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\retrieval\\bm25_index.py
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\retrieval\\hybrid_searcher.py
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\retrieval\\filters.py
G:\\tiaozhanbei\\newrag\\src\\rag_v2\\retrieval\\context_packer.py
G:\\tiaozhanbei\\newrag\\scripts\\build_bm25_stage3.py
G:\\tiaozhanbei\\newrag\\scripts\\search_hybrid_stage3.py
G:\\tiaozhanbei\\newrag\\evaluate_stage3.py
G:\\tiaozhanbei\\newrag\\experiments\\eval_queries_keyword.jsonl
G:\\tiaozhanbei\\newrag\\artifacts\\stage3\\phase3_report.md
```

## 9. GPU 使用边界

阶段三默认不需要 GPU：

- BM25 建索引；
- hybrid 检索；
- metadata filter；
- context pack；
- 阶段三评测。

只有在后续阶段四引入 Cross-Encoder rerank，或者阶段五做更大规模 query 批量实验时，才建议优先使用 GPU。

## 10. 阶段三冻结结论

阶段三已经形成闭环：

```text
Qdrant dense retrieval
+ BM25 sparse retrieval
+ RRF fusion
+ metadata filter
+ context packing
+ stage3 evaluation
```

下一阶段建议进入：

```text
阶段四：Cross-Encoder rerank
```

## 11. 简历表述建议

> 在 Qdrant dense 检索基础上，自研本地 BM25 稀疏召回与 RRF 融合策略，构建 dense+sparse 混合检索链路；支持 source_file 过滤、关键词/条款类问题精确匹配，并实现 section 内邻接 chunk 扩展、相邻 chunk 合并和 token budget 控制，将检索结果组装为可直接供后续 Rerank/Agent 使用的 evidence pack。
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    path = generate_phase3_report()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
