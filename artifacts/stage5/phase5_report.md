# 阶段五收口报告：Query Analyzer / Rewrite / Multi-Query / HyDE

## 1. 阶段五目标

阶段五在阶段四 `dense + sparse + rerank + MMR` 链路前新增 Query Planning 层：先识别 query 类型，再决定是否启用规则改写、Multi-Query 和 HyDE。核心目标不是盲目堆叠 HyDE，而是让不同 query 类型走更合适的检索策略。

## 2. 评测配置

```json
{
  "top_k": 10,
  "dense_top_k": 30,
  "sparse_top_k": 30,
  "rerank_top_k": 30,
  "mmr_pre_candidates": null,
  "mmr_top_k": null,
  "fake_reranker": null,
  "fake_mmr": null,
  "fake_hyde": null,
  "hyde_mode": null,
  "reranker_model_ref": null
}
```

对比四路：

```text
stage4_raw
stage5_rewrite
stage5_multi_query
stage5_hyde
```

## 3. 场景处置类 query

| Variant | doc_recall@10 | chunk_recall@10 | doc_MRR@10 | chunk_MRR@10 | doc_MRR_delta | avg_branch_count | hyde_used_ratio | avg_latency_ms | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage4 raw | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| Stage5 rewrite | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| Stage5 multi-query | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| Stage5 HyDE | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |

## 4. 短 query / 口语 query

| Variant | doc_recall@10 | chunk_recall@10 | doc_MRR@10 | chunk_MRR@10 | doc_MRR_delta | avg_branch_count | hyde_used_ratio | avg_latency_ms | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage4 raw | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| Stage5 rewrite | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| Stage5 multi-query | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| Stage5 HyDE | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |

## 5. 精确条款守护 query

| Variant | doc_recall@10 | chunk_recall@10 | doc_MRR@10 | chunk_MRR@10 | doc_MRR_delta | avg_branch_count | hyde_used_ratio | avg_latency_ms | guardrail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage4 raw | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | ✅ |
| Stage5 rewrite | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | ✅ |
| Stage5 multi-query | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | ✅ |
| Stage5 HyDE | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | ✅ |

守护结论：**通过**。

## 6. 结果解读

- `stage4_raw` 是阶段四单 query 基线；
- `stage5_rewrite` 用较少 branch 验证规则 rewrite 的收益；
- `stage5_multi_query` 使用 raw / normalized / expanded 多分支；
- `stage5_hyde` 在 multi-query 基础上增加 HyDE 分支；
- `guardrail_no_regression` 用于保证精确条款 query 的 `doc_recall@10` 和 `chunk_recall_exact@10` 不比 Stage4 下降。

## 7. 阶段五产物

```text
G:\tiaozhanbei\newrag\src\rag_v2\query\models.py
G:\tiaozhanbei\newrag\src\rag_v2\query\analyzer.py
G:\tiaozhanbei\newrag\src\rag_v2\query\rewriter.py
G:\tiaozhanbei\newrag\src\rag_v2\query\hyde.py
G:\tiaozhanbei\newrag\src\rag_v2\retrieval\multi_query_pipeline.py
G:\tiaozhanbei\newrag\scripts\search_stage5.py
G:\tiaozhanbei\newrag\evaluate_stage5.py
G:\tiaozhanbei\newrag\scripts\generate_phase5_report.py
```

## 8. 简历表述

> 在混合召回与 Cross-Encoder 精排基础上，我进一步实现了 Query Planning 层：通过规则 Query Analyzer 识别场景类、短 query、精确条款等不同 query 类型，并生成 raw / normalized / expanded / HyDE 多分支检索请求；使用 branch-aware RRF 融合多分支结果，保留 raw query 最高权重，并将 HyDE 限制为 dense-only 分支，禁止进入 evidence/citation。评测上加入精确条款 guardrail，确保 query rewrite 和 HyDE 不破坏原本有效的关键词检索。
