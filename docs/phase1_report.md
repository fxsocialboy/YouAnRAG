# 阶段一收口报告：数据管道与索引正确性修复

## 1. 阶段一目标

阶段一聚焦修复原 RAG 数据管道与索引构建中的基础问题，不引入 Qdrant、BM25、reranker、HyDE 等额外变量。核心目标是让被索引文本变得结构正确、长度受控、可定位、可复现。

## 2. 已完成的小阶段

| 小阶段 | 内容 | 状态 |
|---|---|---|
| 1.1 | V2 最小工程骨架、配置与 schema | 完成 |
| 1.2 | 保守 Markdown 清洗 normalizer | 完成 |
| 1.3 | Markdown 标题路径解析 | 完成 |
| 1.4 | Token 分块、overlap、hard max | 完成 |
| 1.5 | Contextual Prefix 与 metadata 输出 | 完成 |
| 1.6 | BGE 编码与 Stage1 FAISS 重建 | 完成 |
| 1.7 | Stage1 检索与 legacy baseline 对比 | 完成 |
| 1.8 | 阶段一冻结与报告 | 完成 |

## 3. 解决的 legacy 问题

1. 修复原 `clean_markdown` 删除所有空白导致的英文粘连和结构丢失问题；
2. 消除 67K 字符超长 chunk，所有 Stage1 chunk 均受 hard max 控制；
3. 引入 Markdown 标题路径 `section_path`，让 chunk 具备章节上下文；
4. 区分 `content` 与 `embedding_text`，避免人工 prefix 污染引用原文；
5. 修复 BGE instruction 方向：passage 不额外加 instruction，query 侧保留 instruction 开关；
6. 封装 FAISS store，增加 `index.ntotal == len(metadata)` 校验；
7. 增加自然语言 query 下的 legacy vs Stage1 对比报告。

## 4. Stage1 数据质量摘要

```json
{
  "total_chunks": 6269,
  "indexable_chunks": 6269,
  "unique_source_files": 51,
  "max_tokens": 448,
  "avg_tokens": 258.96,
  "max_chars": 2066,
  "avg_chars": 276.55,
  "over_hard_max_chunks": 0,
  "missing_section_path_chunks": 0,
  "missing_section_path_ratio": 0.0,
  "duplicate_chunks": 69,
  "duplicate_ratio": 0.011,
  "hard_max_tokens": 448,
  "source_markdown_dir": "/root/autodl-tmp/YouAnRAG/legacy_snapshot/RAG/final_mds",
  "markdown_files": 51,
  "input_blocks": 13287,
  "indexable_blocks": 9697,
  "chunk_params": {
    "target_tokens": 280,
    "soft_max_tokens": 360,
    "hard_max_tokens": 448,
    "overlap_tokens": 50,
    "min_tokens": 40
  }
}
```

关键结论：

- 文档数：51；
- chunk 总数：6269；
- 最大 token 数：448；
- 超过 hard max 的 chunk 数：0；
- 缺失章节路径的 chunk 数：0；
- 重复率：0.011。

## 5. Stage1 FAISS 构建摘要

```json
{
  "metadata_path": "/root/autodl-tmp/YouAnRAG/artifacts/stage1/chunk_metadata.json",
  "index_path": "/root/autodl-tmp/YouAnRAG/artifacts/stage1/faiss_index.index",
  "vector_count": 6269,
  "dimension": 1024,
  "index_ntotal": 6269,
  "metadata_count": 6269,
  "model_path": "/root/autodl-tmp/YouAnRAG/models/bge-large-zh-v1.5",
  "batch_size": 64,
  "device": "cuda",
  "dtype": "float32",
  "max_length": 512,
  "passage_instruction": false,
  "use_query_instruction": true,
  "elapsed_seconds": 69.66
}
```

关键结论：

- 向量条数：6269；
- 向量维度：1024；
- FAISS `index_ntotal`：6269；
- metadata 数量：6269；
- 构建设备：cuda；
- 构建耗时：69.66 秒。

## 6. Legacy vs Stage1 评测摘要

### Legacy

```json
{
  "query_count": 25,
  "top_k": 10,
  "chunk_recall@5": 0.2,
  "chunk_recall@10": 0.2,
  "doc_recall@10": 0.96,
  "total_seconds": 4.42,
  "model_path": "G:\\tiaozhanbei\\Youan-AI-main\\youan-multiagent\\multi_agent_server\\app\\RAG\\bge-large-zh-v1.5",
  "index_path": "G:\\tiaozhanbei\\newrag\\legacy_snapshot\\RAG\\faiss_index.index",
  "metadata_path": "G:\\tiaozhanbei\\newrag\\legacy_snapshot\\RAG\\chunk_metadata.json"
}
```

### Stage1

```json
{
  "query_count": 25,
  "top_k": 10,
  "doc_recall@5": 0.96,
  "doc_recall@10": 0.96,
  "chunk_recall_exact_index@5": 0.12,
  "chunk_recall_exact_index@10": 0.12,
  "avg_latency_ms": 214.34,
  "p95_latency_ms": 248.89,
  "total_seconds": 5.36,
  "metric_note": "chunk_recall_exact_index uses legacy qrels and is strict diagnostic only after re-chunking; doc_recall is the primary comparable metric."
}
```

说明：Stage1 已经重新分块，因此旧评测集中的 `relevant_chunk_index` 与 Stage1 chunk 编号不完全等价。阶段一主要使用 `doc_recall@10` 和 Top-10 定性检查作为可比指标；严格 chunk-level qrels 应在后续重新标注。

## 7. 阶段一产物

```text
G:\tiaozhanbei\newrag\artifacts\stage1\chunks.jsonl
G:\tiaozhanbei\newrag\artifacts\stage1\chunk_metadata.json
G:\tiaozhanbei\newrag\artifacts\stage1\chunk_quality_report.json
G:\tiaozhanbei\newrag\artifacts\stage1\faiss_index.index
G:\tiaozhanbei\newrag\artifacts\stage1\faiss_build_report.json
```

评测产物：

```text
G:\tiaozhanbei\newrag\experiments\stage1_baseline_results.json
G:\tiaozhanbei\newrag\experiments\stage1_vs_legacy_report.md
```

## 8. 阶段一冻结结论

阶段一已经形成独立可运行的优化版 RAG 数据与检索基线。后续阶段应以 `artifacts/stage1` 为输入，不再继续扩大阶段一范围。

下一阶段建议：

```text
阶段二：Qdrant Local 接入与文档自动更新
```

阶段二可以复用当前 Stage1 chunk metadata 和 embedding/index 设计，不需要重新讨论阶段一的分块范围。

## 9. 简历表述建议

> 重构 RAG 数据处理与索引构建链路，修复原系统 Markdown 清洗导致的英文粘连、章节结构丢失和 67K 超长 chunk 问题；实现基于标题路径的 Contextual Prefix、Token 级分块、chunk overlap 与 hard max 控制，重建 BGE + FAISS 索引，并基于自然语言评测集对 legacy 与优化版本进行消融对比。
