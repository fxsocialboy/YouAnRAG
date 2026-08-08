# 阶段二收口报告：Qdrant Local 接入与 Markdown 自动更新

## 1. 阶段二目标

阶段二的核心目标是把阶段一已经冻结的 `FAISS + chunk_metadata` 静态检索链路，升级为 **Qdrant Local + SQLite Registry** 支撑的可更新向量知识库，同时不破坏阶段一产物和检索基线。

本阶段不追求语义效果提升，重点完成三类工程能力：

1. 向量数据库接入；
2. Markdown 文档变化追踪与文档级同步；
3. 与 Stage1 检索结果的一致性验证。

## 2. 已完成的小阶段

| 小阶段 | 内容 | 状态 |
|---|---|---|
| 2.1 | Qdrant 配置、VectorStore 抽象、QdrantStore | 完成 |
| 2.2 | Stage1 FAISS 向量导入 Qdrant | 完成 |
| 2.3 | SQLite DocumentRegistry + FileTracker | 完成 |
| 2.4 | Markdown 文档级同步脚本 | 完成 |
| 2.5 | Qdrant Searcher + 一致性评测 | 完成 |
| 2.6 | 阶段二冻结与报告 | 完成 |

## 3. 阶段二实现的核心能力

### 3.1 向量库接入

- 默认接入 **Qdrant Local**；
- 保留 `QdrantStoreConfig(url=...)`，后续可以迁移到 Qdrant Server；
- 通过 `VectorStore Protocol` 解耦上层检索与底层向量库实现。

### 3.2 文档状态管理

- 使用 SQLite `documents` 表记录文档级 hash、状态、chunk 数；
- 使用 SQLite `chunk_mappings` 表记录 `chunk_id -> qdrant_point_id` 映射；
- 支持 added / modified / deleted / unchanged 变更检测。

### 3.3 文档级同步

- added：解析、分块、embedding、upsert；
- modified：删除旧 points，重建该文档；
- deleted：删除 points，Registry 标记 deleted；
- unchanged：跳过，不做重复写入。

### 3.4 检索迁移一致性验证

- Stage2 Qdrant 检索入口已可用；
- 支持 payload filter；
- 对 25 条 query 做了 Stage1 vs Stage2 Top-10 一致性评测。

## 4. Qdrant 导入摘要

```json
{
  "index_path": "G:\\tiaozhanbei\\newrag\\artifacts\\stage1\\faiss_index.index",
  "metadata_path": "G:\\tiaozhanbei\\newrag\\artifacts\\stage1\\chunk_metadata.json",
  "report_path": "G:\\tiaozhanbei\\newrag\\artifacts\\stage2\\qdrant_import_report.json",
  "qdrant_mode": "local",
  "qdrant_path": "G:\\tiaozhanbei\\newrag\\artifacts\\qdrant_local",
  "qdrant_url": null,
  "collection": "youan_rag_stage2",
  "vector_size": 1024,
  "faiss_ntotal": 6269,
  "metadata_count": 6269,
  "import_limit": null,
  "imported_points": 6269,
  "qdrant_count": 6269,
  "recreate": true,
  "batch_size": 128,
  "health": {
    "backend": "qdrant",
    "mode": "local",
    "collection": "youan_rag_stage2",
    "collection_exists": true,
    "vector_size": 1024,
    "count": 6269,
    "status": "ok"
  },
  "elapsed_seconds": 67.66
}
```

关键结论：

- Qdrant point 数：6269；
- Stage1 metadata 数：6269；
- point 数与 metadata 数一致：True；
- 本步骤不需要 GPU，只复用 Stage1 已有向量。

## 5. Registry 与同步测试摘要

### 5.1 全量 scan-only

```json
{
  "added": 51,
  "modified": 0,
  "deleted": 0,
  "unchanged": 0,
  "total_changed": 51
}
```

说明：当前 Registry 并未对 51 份正式 Markdown 执行全量同步初始化，所以 `--scan-only` 会把现有正式 Markdown 视作待导入集合。这是符合当前阶段目标的：阶段二验证的是**迁移能力与增量同步能力**，而不是强制全量 rebuild。

### 5.2 临时测试文档新增 / 修改 / 删除

新增：

```json
{
  "mode": "sync",
  "source_file_filter": "test_sync.md",
  "changes": {
    "added": 1,
    "modified": 0,
    "deleted": 0,
    "unchanged": 0,
    "total_changed": 1
  },
  "added": [
    {
      "source_file": "test_sync.md",
      "chunks": 1,
      "replace_existing": false
    }
  ],
  "modified": [],
  "deleted": [],
  "qdrant_count": 6270,
  "elapsed_seconds": 0.46
}
```

修改：

```json
{
  "mode": "sync",
  "source_file_filter": "test_sync.md",
  "changes": {
    "added": 0,
    "modified": 1,
    "deleted": 0,
    "unchanged": 0,
    "total_changed": 1
  },
  "added": [],
  "modified": [
    {
      "source_file": "test_sync.md",
      "chunks": 1,
      "replace_existing": true
    }
  ],
  "deleted": [],
  "qdrant_count": 6270,
  "elapsed_seconds": 0.42
}
```

删除：

```json
{
  "mode": "sync",
  "source_file_filter": "test_sync.md",
  "changes": {
    "added": 0,
    "modified": 0,
    "deleted": 1,
    "unchanged": 0,
    "total_changed": 1
  },
  "added": [],
  "modified": [],
  "deleted": [
    {
      "source_file": "test_sync.md",
      "deleted": true
    }
  ],
  "qdrant_count": 6269,
  "elapsed_seconds": 0.13
}
```

清理后再次 scan：

```json
{
  "mode": "scan-only",
  "source_file_filter": "test_sync.md",
  "changes": {
    "added": 0,
    "modified": 0,
    "deleted": 0,
    "unchanged": 0,
    "total_changed": 0
  },
  "added": [],
  "modified": [],
  "deleted": [],
  "unchanged": []
}
```

关键结论：

- 临时 `test_sync.md` 可成功新增到 Qdrant；
- 修改后旧内容被替换，新内容可检索；
- 删除后 Qdrant point 数恢复到 6269；
- 清理后 `test_sync.md` 不再被 scan 发现；
- 当前 Registry 状态：active 文档 0 条，deleted 文档 1 条。

## 6. Stage1 vs Stage2 一致性评测摘要

```json
{
  "query_count": 25,
  "top_k": 10,
  "avg_top10_prefix_match_ratio": 1.0,
  "avg_top10_set_overlap_ratio": 1.0,
  "min_top10_set_overlap_ratio": 1.0,
  "payload_completeness_ratio": 1.0,
  "filter_pass_ratio": 1.0,
  "qdrant_point_count": 6269,
  "metadata_count": 6269,
  "point_count_matches_metadata": true,
  "active_registry_document_count": 0,
  "deleted_registry_document_count": 1,
  "avg_latency_ms": 417.27,
  "p95_latency_ms": 452.93,
  "total_seconds": 17.98,
  "metric_note": "Stage2 focuses on migration consistency between Stage1 FAISS and Qdrant, not recall lift."
}
```

关键结论：

- 25 条 query 的 Top-10 顺序一致率均值：1.0；
- 25 条 query 的 Top-10 集合重合率均值：1.0；
- payload 完整率：1.0；
- filter 正确率：1.0；
- Qdrant point 数与 metadata 数完全对齐：True。

结论：阶段二已经实现了 **FAISS -> Qdrant 的平滑迁移**，且没有引入检索结果漂移。

## 7. Qdrant Local 与 Qdrant Server 的关系

当前实现默认使用：

```text
Qdrant Local + SQLite
```

这样做的好处：

- 本地开发轻量，无需额外服务部署；
- 对简历项目足够完整，已经具备向量数据库、payload filter、upsert/delete、自动更新等核心能力；
- 上层代码通过 `QdrantStoreConfig` 和 `VectorStore Protocol` 解耦，后续切换到 Qdrant Server 只需要改配置，不需要重写检索/同步逻辑。

因此简历上可以表述为：

> 基于 Qdrant 设计可更新向量知识库，开发阶段采用 Qdrant Local，保留 Qdrant Server 迁移配置。

## 8. 阶段二产物

```text
G:\tiaozhanbei\newrag\artifacts\qdrant_local
G:\tiaozhanbei\newrag\artifacts\stage2\qdrant_import_report.json
G:\tiaozhanbei\newrag\artifacts\stage2\document_registry.sqlite3
G:\tiaozhanbei\newrag\artifacts\stage2\sync_scan_report.json
G:\tiaozhanbei\newrag\artifacts\stage2\sync_test_add_report.json
G:\tiaozhanbei\newrag\artifacts\stage2\sync_test_modify_report.json
G:\tiaozhanbei\newrag\artifacts\stage2\sync_test_delete_report.json
G:\tiaozhanbei\newrag\artifacts\stage2\sync_test_clean_scan_report.json
G:\tiaozhanbei\newrag\experiments\stage2_consistency_results.json
```

## 9. GPU 使用边界

不需要 GPU：

- Qdrant Local 导入；
- Qdrant 检索；
- `--scan-only` 变更检测；
- 少量临时 Markdown 的同步测试；
- Stage2 一致性评测。

建议使用 GPU：

- 对 51 份 Markdown 做全量重新 embedding；
- 后续如果阶段三 / 阶段四引入更大模型或 reranker 批量评测。

## 10. 阶段二冻结结论

阶段二已经形成闭环：

```text
Stage1 FAISS vectors
→ Qdrant Local
→ SQLite Registry
→ Markdown change sync
→ Qdrant retrieval
→ Stage1/Stage2 consistency evaluation
```

下一阶段建议：

```text
阶段三：混合召回与上下文组装
```

重点进入：

- BM25 / 关键词召回；
- Dense + Sparse 融合（RRF）；
- 邻接 chunk / section 级上下文组装；
- 为 rerank 和 query rewrite 做更强的候选集准备。

## 11. 简历表述建议

> 将原有基于 FAISS 的静态 RAG 检索迁移为基于 Qdrant 的可更新向量知识库，设计 `VectorStore` 抽象与 `QdrantStore`，复用 Stage1 向量完成 FAISS→Qdrant 平滑迁移；基于 SQLite 构建文档级 Registry 与 chunk 映射，支持 Markdown 的新增、修改、删除同步；实现 Qdrant 检索接口与一致性评测，25 条 query 的 Stage1/Stage2 Top-10 检索结果一致率达到 100%。
