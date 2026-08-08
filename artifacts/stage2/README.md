# Stage2 Artifacts

本目录是阶段二冻结产物目录，保存 Qdrant 接入、SQLite Registry 和同步测试相关结果。

## 文件说明

| 文件 | 作用 |
|---|---|
| `qdrant_import_report.json` | Stage1 FAISS 向量导入 Qdrant 的报告 |
| `document_registry.sqlite3` | SQLite 文档状态与 chunk 映射库 |
| `sync_scan_report.json` | 对正式 Markdown 目录执行 `--scan-only` 的结果 |
| `sync_test_add_report.json` | 临时测试文档新增同步报告 |
| `sync_test_modify_report.json` | 临时测试文档修改同步报告 |
| `sync_test_delete_report.json` | 临时测试文档删除同步报告 |
| `sync_test_clean_scan_report.json` | 清理临时文件后的最终 scan 报告 |

## 当前摘要

```json
{
  "qdrant_point_count": 6269,
  "metadata_count": 6269,
  "point_count_matches_metadata": true,
  "avg_top10_prefix_match_ratio": 1.0,
  "payload_completeness_ratio": 1.0,
  "filter_pass_ratio": 1.0,
  "active_registry_document_count": 0,
  "deleted_registry_document_count": 1
}
```

## 使用方式

扫描变化：

```powershell
python G:\tiaozhanbei\newrag\scripts\sync_qdrant_stage2.py --scan-only
```

按文件同步：

```powershell
python G:\tiaozhanbei\newrag\scripts\sync_qdrant_stage2.py --sync --source-file test_sync.md --device cpu --batch-size 4
```

Qdrant 检索：

```powershell
python G:\tiaozhanbei\newrag\scripts\search_qdrant_stage2.py "IV级气象灾害应急响应一般由谁启动？" --top-k 3 --device cpu
```

一致性评测：

```powershell
python G:\tiaozhanbei\newrag\evaluate_stage2.py --top-k 10 --device cpu
```

## 冻结规则

阶段二收口后：

1. 不手工改写 `artifacts\qdrant_local` 内部文件；
2. 不手工改写 `document_registry.sqlite3`；
3. 如果要重新做全量导入，可运行：

```powershell
python G:\tiaozhanbei\newrag\scripts\import_stage1_to_qdrant.py --recreate
```

4. 如果要全量重建所有 Markdown 的 embedding，同步阶段建议使用 GPU。
