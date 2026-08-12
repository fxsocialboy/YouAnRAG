# YouAn RAG V2 阶段七最终评测报告

> 本报告由冻结评测结果自动生成。随机鲁棒性集没有人工相关性标签，不报告 Recall。

## 1. 评测范围

- 人工标注集：2 条；Legacy 与 V2 Full 比较检索指标。
- 固定种子随机集：2 条；评测真实回答、降级、引用、Judge 质量和延迟。
- V2 Full：BGE + Qdrant + BM25/RRF + Cross-Encoder + MMR + Query Rewrite/Multi-Query/条件 HyDE + DeepSeek。
- DeepSeek 同时作为 Composer 和 Judge，指标存在同模型偏差；引用映射和 fallback 指标为确定性指标。

## 2. 标注集 Legacy vs V2 Full 检索指标

| 指标 | Legacy | V2 Full | 差值(V2-Legacy) |
|---|---:|---:|---:|
| doc_recall@5 | 1.0000 | 1.0000 | 0.0000 |
| doc_recall@10 | 1.0000 | 1.0000 | 0.0000 |
| chunk_recall@5 | 0.5000 | 0.0000 | -0.5000 |
| chunk_recall@10 | 0.5000 | 0.0000 | -0.5000 |
| doc_mrr@10 | 0.7500 | 1.0000 | 0.2500 |
| chunk_mrr@10 | 0.5000 | 0.0000 | -0.5000 |
| doc_ndcg@10 | 0.8155 | 1.0000 | 0.1845 |
| chunk_ndcg@10 | 0.5000 | 0.0000 | -0.5000 |
| avg_latency_ms | 206.9400 | 50210.4000 | 50003.4600 |
| p50_latency_ms | 206.9400 | 50210.4000 | 50003.4600 |
| p95_latency_ms | 260.3900 | 50566.0100 | 50305.6200 |

## 3. 标注集 V2 答案质量

| 指标 | 结果 |
|---|---:|
| answer_success_rate | 1.0000 |
| fallback_rate | 0.0000 |
| fallback_accuracy | 1.0000 |
| faithfulness | N/A |
| answer_relevancy | N/A |
| citation_correctness | N/A |
| citation_completeness | N/A |
| all_citations_mapped_ratio | 1.0000 |
| avg_latency_ms | 50210.8700 |
| p50_latency_ms | 50210.8700 |
| p95_latency_ms | 50566.5700 |

## 4. 随机鲁棒性集

| 指标 | 结果 |
|---|---:|
| answer_success_rate | 1.0000 |
| fallback_rate | 0.0000 |
| fallback_accuracy | 1.0000 |
| faithfulness | N/A |
| answer_relevancy | N/A |
| citation_correctness | N/A |
| citation_completeness | N/A |
| all_citations_mapped_ratio | 1.0000 |
| avg_latency_ms | 50210.8700 |
| p50_latency_ms | 50210.8700 |
| p95_latency_ms | 50566.5700 |

确定性补充指标：

- 未知引用数量：0；
- 实际使用 DeepSeek Composer 比例：0.0000；
- in-domain answer rate：1.0000；
- OOD fallback accuracy：0.0000。

## 5. 按灾种拆分

| 灾种 | 数量 | Faithfulness | Relevancy | Fallback rate |
|---|---:|---:|---:|---:|
| flood | 1 | N/A | N/A | 0.0000 |
| geological | 1 | N/A | N/A | 0.0000 |

## 6. 低分、降级或失败样例

```json
[
  {
    "query_id": "final_001",
    "query": "四川山区群众发现滑坡迹象后如何实现提前避险？",
    "status": "ok",
    "fallback": false,
    "faithfulness": null,
    "error": null
  },
  {
    "query_id": "final_002",
    "query": "我国为什么需要专门做山洪灾害防治规划？",
    "status": "ok",
    "fallback": false,
    "faithfulness": null,
    "error": null
  },
  {
    "query_id": "final_001",
    "query": "四川山区群众发现滑坡迹象后如何实现提前避险？",
    "status": "ok",
    "fallback": false,
    "faithfulness": null,
    "error": null
  },
  {
    "query_id": "final_002",
    "query": "我国为什么需要专门做山洪灾害防治规划？",
    "status": "ok",
    "fallback": false,
    "faithfulness": null,
    "error": null
  }
]
```

## 7. 结果解释边界

- 知识库仅包含 51 份灾害相关 Markdown，覆盖范围之外的问题应合理降级。
- LLM-as-Judge 分数用于规模化趋势分析，不等同于人工专家审查。
- 最终简历只引用本报告中的真实指标，不把随机集包装成有人工标签的准确率测试。
- 人工抽查记录应另填 `stage7_manual_review.md`，至少抽查总样本的 10%。
