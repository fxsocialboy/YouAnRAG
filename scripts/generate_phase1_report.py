"""Generate phase-1 closing report and artifact README."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_v2.config import get_config


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def generate_phase1_report(project_root: Path = PROJECT_ROOT) -> tuple[Path, Path]:
    cfg = get_config()
    stage_dir = cfg.stage1_artifacts_dir
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    quality = read_json(stage_dir / "chunk_quality_report.json")
    faiss_report = read_json(stage_dir / "faiss_build_report.json")
    stage1_eval = read_json(project_root / "experiments" / "stage1_baseline_results.json")
    legacy_eval = read_json(project_root / "experiments" / "legacy_baseline_results.json")
    s_summary = stage1_eval.get("summary", {})
    l_summary = legacy_eval.get("summary", {})

    phase1_report = docs_dir / "phase1_report.md"
    artifact_readme = stage_dir / "README.md"

    phase1_text = f"""# 阶段一收口报告：数据管道与索引正确性修复

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
{json.dumps(quality, ensure_ascii=False, indent=2)}
```

关键结论：

- 文档数：{quality.get('unique_source_files', 'N/A')}；
- chunk 总数：{quality.get('total_chunks', 'N/A')}；
- 最大 token 数：{quality.get('max_tokens', 'N/A')}；
- 超过 hard max 的 chunk 数：{quality.get('over_hard_max_chunks', 'N/A')}；
- 缺失章节路径的 chunk 数：{quality.get('missing_section_path_chunks', 'N/A')}；
- 重复率：{quality.get('duplicate_ratio', 'N/A')}。

## 5. Stage1 FAISS 构建摘要

```json
{json.dumps(faiss_report, ensure_ascii=False, indent=2)}
```

关键结论：

- 向量条数：{faiss_report.get('vector_count', 'N/A')}；
- 向量维度：{faiss_report.get('dimension', 'N/A')}；
- FAISS `index_ntotal`：{faiss_report.get('index_ntotal', 'N/A')}；
- metadata 数量：{faiss_report.get('metadata_count', 'N/A')}；
- 构建设备：{faiss_report.get('device', 'N/A')}；
- 构建耗时：{faiss_report.get('elapsed_seconds', 'N/A')} 秒。

## 6. Legacy vs Stage1 评测摘要

### Legacy

```json
{json.dumps(l_summary, ensure_ascii=False, indent=2)}
```

### Stage1

```json
{json.dumps(s_summary, ensure_ascii=False, indent=2)}
```

说明：Stage1 已经重新分块，因此旧评测集中的 `relevant_chunk_index` 与 Stage1 chunk 编号不完全等价。阶段一主要使用 `doc_recall@10` 和 Top-10 定性检查作为可比指标；严格 chunk-level qrels 应在后续重新标注。

## 7. 阶段一产物

```text
G:\\tiaozhanbei\\newrag\\artifacts\\stage1\\chunks.jsonl
G:\\tiaozhanbei\\newrag\\artifacts\\stage1\\chunk_metadata.json
G:\\tiaozhanbei\\newrag\\artifacts\\stage1\\chunk_quality_report.json
G:\\tiaozhanbei\\newrag\\artifacts\\stage1\\faiss_index.index
G:\\tiaozhanbei\\newrag\\artifacts\\stage1\\faiss_build_report.json
```

评测产物：

```text
G:\\tiaozhanbei\\newrag\\experiments\\stage1_baseline_results.json
G:\\tiaozhanbei\\newrag\\experiments\\stage1_vs_legacy_report.md
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
"""

    artifact_text = f"""# Stage1 Artifacts

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
{json.dumps({
    'total_chunks': quality.get('total_chunks'),
    'unique_source_files': quality.get('unique_source_files'),
    'max_tokens': quality.get('max_tokens'),
    'over_hard_max_chunks': quality.get('over_hard_max_chunks'),
    'index_ntotal': faiss_report.get('index_ntotal'),
    'metadata_count': faiss_report.get('metadata_count'),
    'dimension': faiss_report.get('dimension'),
}, ensure_ascii=False, indent=2)}
```

## 使用方式

搜索：

```powershell
python G:\\tiaozhanbei\\newrag\\scripts\\search_stage1.py "IV级气象灾害应急响应一般由谁启动？" --top-k 3 --device cpu
```

评测：

```powershell
python G:\\tiaozhanbei\\newrag\\evaluate_stage1.py --top-k 10 --device cpu
```

## 冻结规则

阶段一收口后，不再手动修改本目录产物。若需要变更分块参数或 metadata 字段，应重新运行：

```powershell
python G:\\tiaozhanbei\\newrag\\scripts\\build_chunks_stage1.py
python G:\\tiaozhanbei\\newrag\\scripts\\build_faiss_stage1.py --batch-size 64 --device cuda
```

第二条涉及全量 BGE embedding，建议使用 GPU 服务器执行。
"""

    phase1_report.write_text(phase1_text, encoding="utf-8")
    artifact_readme.write_text(artifact_text, encoding="utf-8")
    return phase1_report, artifact_readme


def main() -> None:
    phase1_report, artifact_readme = generate_phase1_report()
    print(f"wrote {phase1_report}")
    print(f"wrote {artifact_readme}")


if __name__ == "__main__":
    main()
