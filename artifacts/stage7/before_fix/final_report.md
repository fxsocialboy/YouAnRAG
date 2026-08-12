# YouAn RAG V2 阶段七最终评测报告

> 本报告由冻结评测结果自动生成。随机鲁棒性集没有人工相关性标签，不报告 Recall。

## 1. 评测范围

- 人工标注集：25 条；Legacy 与 V2 Full 比较检索指标。
- 固定种子随机集：120 条；评测真实回答、降级、引用、Judge 质量和延迟。
- V2 Full：BGE + Qdrant + BM25/RRF + Cross-Encoder + MMR + Query Rewrite/Multi-Query/条件 HyDE + DeepSeek。
- DeepSeek 同时作为 Composer 和 Judge，指标存在同模型偏差；引用映射和 fallback 指标为确定性指标。

## 2. 标注集 Legacy vs V2 Full 检索指标

| 指标 | Legacy | V2 Full | 差值(V2-Legacy) |
|---|---:|---:|---:|
| doc_recall@5 | 0.8913 | 0.9348 | 0.0435 |
| doc_recall@10 | 0.8913 | 0.9348 | 0.0435 |
| chunk_recall@5 | 0.5000 | 0.6304 | 0.1304 |
| chunk_recall@10 | 0.5870 | 0.6304 | 0.0434 |
| doc_mrr@10 | 0.8696 | 0.8768 | 0.0072 |
| chunk_mrr@10 | 0.3686 | 0.3732 | 0.0046 |
| doc_ndcg@10 | 0.8641 | 0.8802 | 0.0161 |
| chunk_ndcg@10 | 0.4092 | 0.4336 | 0.0244 |
| avg_latency_ms | 37.2500 | 1927.2500 | 1890.0000 |
| p50_latency_ms | 8.1200 | 2203.3600 | 2195.2400 |
| p95_latency_ms | 9.3700 | 2746.7300 | 2737.3600 |

## 3. 标注集 V2 答案质量

| 指标 | 结果 |
|---|---:|
| answer_success_rate | 1.0000 |
| fallback_rate | 0.1600 |
| fallback_accuracy | 0.7600 |
| faithfulness | 0.9762 |
| answer_relevancy | 0.8571 |
| citation_correctness | 1.0000 |
| citation_completeness | 0.9048 |
| all_citations_mapped_ratio | 1.0000 |
| avg_latency_ms | 7146.4800 |
| p50_latency_ms | 7135.1900 |
| p95_latency_ms | 11800.3400 |

## 4. 随机鲁棒性集

| 指标 | 结果 |
|---|---:|
| answer_success_rate | 1.0000 |
| fallback_rate | 0.2333 |
| fallback_accuracy | 0.0500 |
| faithfulness | 0.8870 |
| answer_relevancy | 0.6957 |
| citation_correctness | 0.9984 |
| citation_completeness | 0.7677 |
| all_citations_mapped_ratio | 1.0000 |
| avg_latency_ms | 8046.5800 |
| p50_latency_ms | 7514.9100 |
| p95_latency_ms | 13880.3500 |

确定性补充指标：

- 未知引用数量：0；
- 实际使用 DeepSeek Composer 比例：1.0000；
- in-domain answer rate：0.7300；
- OOD fallback accuracy：0.0500。

## 5. 按灾种拆分

| 灾种 | 数量 | Faithfulness | Relevancy | Fallback rate |
|---|---:|---:|---:|---:|
| comprehensive | 34 | 0.8869 | 0.8235 | 0.5000 |
| earthquake | 4 | 0.8690 | 0.5333 | 0.2500 |
| flood | 18 | 0.9227 | 0.9250 | 0.1111 |
| geological | 21 | 0.9363 | 0.9167 | 0.1429 |
| meteorological | 23 | 0.9679 | 0.8895 | 0.1739 |
| out_of_domain | 20 | 0.7325 | 0.0105 | 0.0500 |

## 6. 低分、降级或失败样例

```json
[
  {
    "query_id": "final_006",
    "query": "气象灾害应急响应命令一般由谁签发？",
    "status": "ok",
    "fallback": true,
    "faithfulness": null,
    "error": null
  },
  {
    "query_id": "final_014",
    "query": "IV级 应急响应 命令 副局长 签发",
    "status": "ok",
    "fallback": true,
    "faithfulness": null,
    "error": null
  },
  {
    "query_id": "final_019",
    "query": "自然灾害 情况 统计调查制度 统计法 第七条",
    "status": "ok",
    "fallback": true,
    "faithfulness": null,
    "error": null
  },
  {
    "query_id": "final_020",
    "query": "广东省 自然灾害救助 省减灾委员会 组织指挥体系",
    "status": "ok",
    "fallback": true,
    "faithfulness": null,
    "error": null
  },
  {
    "query_id": "random_004",
    "query": "如果工业园区面临地震，而且连续降雨已经两天，应急处置顺序怎么安排？",
    "status": "ok",
    "fallback": true,
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
