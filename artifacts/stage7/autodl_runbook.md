# Stage 7 AutoDL 最终评测操作手册

## 零、本地生成上传包

在 Windows PowerShell 执行：

```powershell
cd G:\tiaozhanbei\newrag
powershell -ExecutionPolicy Bypass -File scripts\package_stage7_autodl.ps1
```

生成：

```text
G:\tiaozhanbei\newrag_stage7_code.tar.gz
```

该压缩包包含代码、51 份 Markdown、Stage1/Stage3 索引和评测集，但不包含模型及 Qdrant Local 数据库。这样既避免重复上传大模型，又避免 Windows/Linux 中文文件名乱码。

把压缩包上传至：

```text
/root/autodl-tmp/newrag_stage7_code.tar.gz
```

如果服务器现有 `/root/autodl-tmp/YouAnRAG/models` 中两个模型都完整，可直接复用。进入服务器后覆盖解压代码：

```bash
cd /root/autodl-tmp
mkdir -p YouAnRAG
tar -xzf newrag_stage7_code.tar.gz -C YouAnRAG
cd YouAnRAG
```

因为上传包排除了 `models/`，覆盖解压不会删除服务器已有模型。

## 一、服务器需要具备的目录

```text
/root/autodl-tmp/YouAnRAG/
├── evaluate_stage7.py
├── requirements-stage7-autodl.txt
├── models/
│   ├── bge-large-zh-v1.5/
│   └── bge-reranker-base/
├── artifacts/
│   ├── stage1/faiss_index.index
│   ├── stage1/chunk_metadata.json
│   ├── stage3/bm25_index.json
│   └── qdrant_local/       # 可不存在，运行脚本会自动重建
├── experiments/
│   ├── eval_queries_final_labeled.jsonl
│   └── eval_queries_final_random.jsonl
└── legacy_snapshot/RAG/
    ├── faiss_index.index
    └── chunk_metadata.json
```

模型必须分别位于上述两个独立目录，不要将两个模型文件混到同一个目录。

## 二、安装依赖

使用 AutoDL 的 `PyTorch 2.1.0 / Python 3.10 / Ubuntu 22.04 / CUDA 12.1` 镜像：

```bash
cd /root/autodl-tmp/YouAnRAG
python -m pip install -r requirements-stage7-autodl.txt
python -m pip install -e . --no-deps
```

不要重新安装 torch。验证环境：

```bash
python - <<'PY'
import torch, transformers, sentence_transformers, qdrant_client
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('sentence-transformers:', sentence_transformers.__version__)
print('cuda:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0))
PY
```

## 三、先跑两条本地链路检查

```bash
python evaluate_stage7.py \
  --dataset labeled --backend v2 --limit 2 \
  --device cuda --reranker-device cuda \
  --embedding-model-path models/bge-large-zh-v1.5 \
  --reranker-model-path models/bge-reranker-base \
  --composer-mode deepseek --hyde-mode deepseek \
  --out artifacts/stage7/smoke_labeled_v2.json --fresh
```

检查 JSON 中：

- `status` 应为 `ok`；
- `composer_mode` 应为 `deepseek`；
- `judge` 不应为空；
- `unknown_citation_ids` 应为空；
- `trace.query_plan` 和 `trace.branches` 应存在。

## 四、运行完整评测

```bash
chmod +x scripts/run_stage7_autodl.sh
bash scripts/run_stage7_autodl.sh
```

完整流程依次执行：

1. 25 条 Legacy 标注集检索；
2. 25 条 V2 Full 标注集检索、DeepSeek 回答和 Judge；
3. 120 条随机鲁棒性集 V2 Full 回答和 Judge；
4. 自动生成最终指标、Markdown 报告和人工抽查表；
5. 打包为 `artifacts/stage7_results.tar.gz`。

## 五、断线或 API 临时失败

直接重新执行同一条命令：

```bash
bash scripts/run_stage7_autodl.sh
```

输出文件按 query 即时原子落盘。状态为 `ok` 的 query 会跳过，`partial/error` 会因为脚本携带 `--retry-errors` 自动重试，不会从头重新消耗 API。

可以观察进度：

```bash
tail -f artifacts/stage7/logs/03_random_v2.log
```

## 六、结果检查

```bash
python - <<'PY'
import json
from pathlib import Path
for name in ['final_labeled_legacy_eval.json', 'final_labeled_v2_eval.json', 'final_random_v2_eval.json']:
    d=json.loads((Path('artifacts/stage7')/name).read_text(encoding='utf-8'))
    print('\n', name)
    print(json.dumps(d['summary'], ensure_ascii=False, indent=2))
PY
```

重点检查：

- `error_count` 和 `partial_count`；
- `actual_deepseek_count`；
- `unknown_citation_count`；
- `faithfulness`、`answer_relevancy`；
- `fallback_accuracy`；
- `avg/p50/p95 latency`。

## 七、人工抽查

打开：

```text
artifacts/stage7/stage7_manual_review.md
```

脚本固定抽取 15 条，覆盖高分、低分、fallback、错误和普通样本。依次填写：

- 人工结论；
- 问题分类；
- 备注。

填写完成后重新打包：

```bash
tar -czf artifacts/stage7_results.tar.gz artifacts/stage7
```

## 八、下载回本地

从 AutoDL 文件管理器下载：

```text
/root/autodl-tmp/YouAnRAG/artifacts/stage7_results.tar.gz
```

下载到：

```text
G:\tiaozhanbei\stage7_results.tar.gz
```
