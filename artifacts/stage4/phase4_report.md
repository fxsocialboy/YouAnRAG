# 阶段四收口报告：Cross-Encoder 精排与 MMR 多样性控制

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
G:\tiaozhanbei\newrag\experiments\stage4_rerank_eval.json
```

本次评测配置：

```json
{
  "top_k": 10,
  "dense_top_k": 30,
  "sparse_top_k": 30,
  "rerank_top_k": 30,
  "mmr_pre_candidates": 20,
  "mmr_top_k": 10,
  "fake_reranker": false,
  "fake_mmr": false,
  "mmr_lambda": 0.7,
  "reranker_model_ref": "models/bge-reranker-base"
}
```

对比四路：

```text
hybrid
hybrid + rerank
hybrid + MMR
hybrid + rerank + MMR
```

## 4. 自然语言 query 结果

| Variant | doc_recall@10 | chunk_recall@10 | doc_MRR@10 | chunk_MRR@10 | doc_MRR_delta | avg_doc_rank_shift | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 1.0 | 0.7429 | 0.9238 | 0.5662 | 0.0 | 0.0 | 196.52 |
| hybrid + rerank | 1.0 | 0.6857 | 0.9629 | 0.4398 | 0.0391 | -0.1429 | 303.68 |
| hybrid + MMR | 1.0 | 0.7429 | 0.931 | 0.5638 | 0.0072 | -0.0857 | 321.38 |
| hybrid + rerank + MMR | 1.0 | 0.6857 | 0.9629 | 0.4448 | 0.0391 | -0.1429 | 399.21 |

## 5. 关键词 / 条款 query 结果

| Variant | doc_recall@10 | chunk_recall@10 | doc_MRR@10 | chunk_MRR@10 | doc_MRR_delta | avg_doc_rank_shift | avg_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 1.0 | 0.9333 | 1.0 | 0.7583 | 0.0 | 0.0 | 154.68 |
| hybrid + rerank | 1.0 | 0.7333 | 0.9222 | 0.39 | -0.0778 | 0.2 | 243.23 |
| hybrid + MMR | 1.0 | 0.9333 | 1.0 | 0.7583 | 0.0 | 0.0 | 324.36 |
| hybrid + rerank + MMR | 1.0 | 0.7333 | 0.9222 | 0.4206 | -0.0778 | 0.2 | 387.96 |

## 6. 结果解读

- `doc_recall@10` / `chunk_recall_exact@10` 说明相关证据是否进入 Top-10；
- `MRR@10` 说明相关证据排在多靠前，是阶段四最核心指标；
- `rank_shift` 为负数表示相关文档或 chunk 被前移；
- `avg_latency_ms` 用于衡量 rerank/MMR 带来的额外代价。

如果 `fake_reranker=true`，本报告只说明链路可跑通，不代表真实 reranker 效果。真实收口建议使用本地轻量真实 smoke 或 AutoDL 完整评测结果覆盖本文件。

## 7. 代表性 rank 前移样例

- `r003` / `四川山区群众发现滑坡迹象后如何实现提前避险？`：doc_rank_shift=-2，chunk_rank_shift=None
- `r001` / `如果想了解某年大陆地震造成的总体损失，应该查哪类资料？`：doc_rank_shift=-1，chunk_rank_shift=None
- `r002` / `2016年大陆5级以上地震和灾害损失情况在哪里能看到？`：doc_rank_shift=-1，chunk_rank_shift=-1


## 8. 阶段四产物

```text
G:\tiaozhanbei\newrag\src\rag_v2\retrieval\reranker.py
G:\tiaozhanbei\newrag\src\rag_v2\retrieval\mmr.py
G:\tiaozhanbei\newrag\src\rag_v2\retrieval\rerank_pipeline.py
G:\tiaozhanbei\newrag\scripts\search_rerank_stage4.py
G:\tiaozhanbei\newrag\evaluate_stage4.py
G:\tiaozhanbei\newrag\scripts\generate_phase4_report.py
G:\tiaozhanbei\newrag\artifacts\stage4\phase4_report.md
```

## 9. GPU / AutoDL 建议

本地 CPU 可以跑：

```powershell
python evaluate_stage4.py --fake-reranker --fake-mmr --limit 2 --device cpu
python scripts\search_rerank_stage4.py "IV????????????????" --top-k 5 --device cpu --reranker-model-path G:\tiaozhanbei\newrag\models\bge-reranker-base --no-mmr
```

完整真实评测建议在 GPU / AutoDL 上跑：

```bash
python evaluate_stage4.py --device cuda --reranker-device cuda --reranker-model-path models/bge-reranker-base
```

## 10. 阶段四简历表述

> 在 dense+sparse 混合召回的 RRF 粗排基础上，引入 Cross-Encoder reranker 对 query-passage pair 做联合建模精排，并使用 MMR 在相关性与多样性之间做折中；设计 `mmr_score > rerank_score > fusion_score` 的三级降级排序，使精排和多样性结果真正进入最终 evidence pack；通过 MRR@10、rank_shift、recall 和 latency 对 hybrid/rerank/MMR 进行消融评测，同时保留开关和异常 fallback，保证可回退到阶段三 hybrid 链路。
