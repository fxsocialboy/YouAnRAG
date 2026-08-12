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
| chunk_recall@10 | 0.5870 | 0.7609 | 0.1739 |
| doc_mrr@10 | 0.8696 | 0.8913 | 0.0217 |
| chunk_mrr@10 | 0.3686 | 0.4639 | 0.0953 |
| doc_ndcg@10 | 0.8641 | 0.8916 | 0.0275 |
| chunk_ndcg@10 | 0.4092 | 0.5305 | 0.1213 |
| avg_latency_ms | 36.7700 | 2194.9500 | 2158.1800 |
| p50_latency_ms | 8.1500 | 2417.2900 | 2409.1400 |
| p95_latency_ms | 14.2400 | 3479.6500 | 3465.4100 |

## 3. 标注集 V2 答案质量

| 指标 | 结果 |
|---|---:|
| answer_success_rate | 1.0000 |
| fallback_rate | 0.0800 |
| fallback_accuracy | 1.0000 |
| faithfulness | 1.0000 |
| answer_relevancy | 1.0000 |
| citation_correctness | 1.0000 |
| citation_completeness | 0.9913 |
| all_citations_mapped_ratio | 1.0000 |
| avg_latency_ms | 7601.2500 |
| p50_latency_ms | 7489.1200 |
| p95_latency_ms | 13788.4400 |

## 4. 随机鲁棒性集

| 指标 | 结果 |
|---|---:|
| answer_success_rate | 1.0000 |
| fallback_rate | 0.3500 |
| fallback_accuracy | 1.0000 |
| faithfulness | 0.9702 |
| answer_relevancy | 0.9462 |
| citation_correctness | 0.9968 |
| citation_completeness | 0.9271 |
| all_citations_mapped_ratio | 1.0000 |
| avg_latency_ms | 9063.2800 |
| p50_latency_ms | 8075.2700 |
| p95_latency_ms | 21220.6600 |

确定性补充指标：

- 未知引用数量：0；
- 实际使用 DeepSeek Composer 比例：0.7833；
- in-domain answer rate：0.7800；
- OOD fallback accuracy：1.0000。

## 5. 按灾种拆分

| 灾种 | 数量 | Faithfulness | Relevancy | Fallback rate |
|---|---:|---:|---:|---:|
| comprehensive | 34 | 0.9776 | 0.9483 | 0.1471 |
| earthquake | 4 | 1.0000 | 1.0000 | 0.2500 |
| flood | 18 | 0.9674 | 0.9385 | 0.2778 |
| geological | 21 | 0.9737 | 0.9316 | 0.0952 |
| meteorological | 23 | 0.9464 | 0.9571 | 0.3913 |
| out_of_domain | 20 | N/A | N/A | 1.0000 |

## 6. 低分、降级或失败样例

```json
[
  {
    "query_id": "random_021",
    "query": "如果养老院面临台风，而且道路部分积水，应急处置顺序怎么安排？",
    "status": "partial",
    "fallback": false,
    "faithfulness": 0.6666666666666666,
    "error": null
  },
  {
    "query_id": "random_028",
    "query": "如果乡镇卫生院面临泥石流，而且现场人员缺少专业设备，应急处置顺序怎么安排？",
    "status": "partial",
    "fallback": false,
    "faithfulness": 1.0,
    "error": null
  },
  {
    "query_id": "random_063",
    "query": "请问天津市的Ⅱ级响应由谁决定，相关解除条件是什么？",
    "status": "partial",
    "fallback": false,
    "faithfulness": 1.0,
    "error": null
  },
  {
    "query_id": "final_024",
    "query": "如何用量子纠错码提高超导量子计算机的逻辑门保真度？",
    "status": "ok",
    "fallback": true,
    "faithfulness": null,
    "error": null
  },
  {
    "query_id": "final_025",
    "query": "请给我推荐适合训练大语言模型的GPU集群网络拓扑。",
    "status": "ok",
    "fallback": true,
    "faithfulness": null,
    "error": null
  }
]
```

## 7. OOD与Fallback决策

```json
{
  "confusion_matrix": {
    "ood_fallback": 22,
    "ood_answered": 0,
    "in_domain_fallback": 22,
    "in_domain_answered": 101
  },
  "ood_false_accept_rate": 0.0,
  "in_domain_false_rejection_rate": 0.1789,
  "fallback_reason_distribution": {
    "insufficient_evidence": 19,
    "out_of_domain": 25
  }
}
```

## 8. Before/After对比

已加载before-fix指标快照，详见final_metrics.json中的before_fix字段。

## 9. 结果解释边界

- 知识库仅包含 51 份灾害相关 Markdown，覆盖范围之外的问题应合理降级。
- LLM-as-Judge 分数用于规模化趋势分析，不等同于人工专家审查。
- 最终简历只引用本报告中的真实指标，不把随机集包装成有人工标签的准确率测试。
- 人工抽查记录应另填 `stage7_manual_review.md`，至少抽查总样本的 10%。
