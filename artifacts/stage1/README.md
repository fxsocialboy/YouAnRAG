# Stage1 Artifacts

本目录是阶段一冻结产物目录，作为后续阶段二 Qdrant Local 接入和自动更新开发的输入。

## 文件说明

| 文件 | 作用 |
|---|---|
| `chunks.jsonl` | 一行一个 Stage1 chunk，适合流式处理 |
| `chunk_metadata.json` | Stage1 chunk metadata 数组，后续 FAISS/Qdrant 检索使用 |
| `chunk_quality_report.json` | 分块质量报告 |
| `faiss_index.index` | 基于 `embedding_text` 构建的 Stage1 FAISS 索引 |
| `faiss_build_report.json` | FAISS 构建报告 |

## 当前摘要

```json
{
  "total_chunks": 6269,
  "unique_source_files": 51,
  "max_tokens": 448,
  "over_hard_max_chunks": 0,
  "index_ntotal": 6269,
  "metadata_count": 6269,
  "dimension": 1024
}
```

## 使用方式

搜索：

```powershell
python G:\tiaozhanbei\newrag\scripts\search_stage1.py "IV级气象灾害应急响应一般由谁启动？" --top-k 3 --device cpu
```

评测：

```powershell
python G:\tiaozhanbei\newrag\evaluate_stage1.py --top-k 10 --device cpu
```

## 冻结规则

阶段一收口后，不再手动修改本目录产物。若需要变更分块参数或 metadata 字段，应重新运行：

```powershell
python G:\tiaozhanbei\newrag\scripts\build_chunks_stage1.py
python G:\tiaozhanbei\newrag\scripts\build_faiss_stage1.py --batch-size 64 --device cuda
```

第二条涉及全量 BGE embedding，建议使用 GPU 服务器执行。
