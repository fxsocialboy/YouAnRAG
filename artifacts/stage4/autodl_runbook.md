# Stage4 AutoDL 真实评测运行说明

本文件用于在 AutoDL/GPU 服务器上跑阶段四真实 reranker 完整评测。

## 0. 本次评测集说明

阶段四准备了两套评测集：

1. 原小评测集：

```text
experiments/eval_queries.jsonl                    # 25 条自然语言 query
experiments/eval_queries_keyword.jsonl            # 10 条关键词 query
```

2. 阶段四扩展真实评测集：

```text
experiments/eval_queries_stage4_realistic.jsonl          # 35 条更随机的自然语言 query
experiments/eval_queries_stage4_keyword_realistic.jsonl  # 15 条关键词/条款 query
```

建议 AutoDL 正式评测优先使用第二套扩展集。它覆盖地震、气象、洪涝、地质灾害、风险普查、灾情统计、气候变化等不同类型文档。

当前评测指标包括：

```text
doc_recall@10
chunk_recall_exact@10
doc_mrr@10
chunk_mrr_exact@10
doc_rank_shift_vs_hybrid
chunk_rank_shift_vs_hybrid
packed_nonempty_ratio
avg_packed_token_ratio
avg_latency_ms
```

说明：

- `doc_recall@10`：相关文档是否进入 Top-10；
- `chunk_recall_exact@10`：标注 chunk 是否进入 Top-10，比较严格；
- `MRR@10`：相关文档/chunk 排得越靠前，分数越高；
- `rank_shift_vs_hybrid`：负数表示比 Stage3 hybrid 更靠前；
- `avg_latency_ms`：评估 reranker/MMR 的额外延迟。

## 1. 进入项目

```bash
cd /root/autodl-tmp/YouAnRAG
# 如果仓库已有更新：
git pull
```

## 2. 检查必要产物

```bash
ls artifacts/stage1/chunk_metadata.json
ls artifacts/stage3/bm25_index.json
ls artifacts/qdrant_local
ls models/bge-large-zh-v1.5
```

reranker 模型建议放在：

```bash
models/bge-reranker-base
```

检查：

```bash
ls models/bge-reranker-base
```

如果本地已经下载模型，可以压缩上传到 AutoDL 后解压到 `models/bge-reranker-base`。

## 3. 跑单元测试

```bash
python -m pytest tests -q --basetemp=artifacts/pytest_tmp
python scripts/check_legacy_hash.py
```

## 4. 先跑真实 reranker 单条 smoke

```bash
python scripts/search_rerank_stage4.py "IV级气象灾害应急响应一般由谁启动？" \
  --top-k 5 \
  --device cuda \
  --reranker-device cuda \
  --reranker-model-path models/bge-reranker-base \
  --no-mmr
```

确认输出中：

```text
rerank_score != null
packed_context 正常
```

## 5. 跑扩展集完整阶段四真实评测

推荐先跑扩展真实评测集：

```bash
python evaluate_stage4.py \
  --queries experiments/eval_queries_stage4_realistic.jsonl \
  --keyword-queries experiments/eval_queries_stage4_keyword_realistic.jsonl \
  --out experiments/stage4_rerank_eval_realistic.json \
  --device cuda \
  --batch-size 64 \
  --reranker-device cuda \
  --reranker-batch-size 32 \
  --reranker-model-path models/bge-reranker-base
```

输出：

```text
experiments/stage4_rerank_eval_realistic.json
```

如果想同时保留默认输出文件，再跑：

```bash
cp experiments/stage4_rerank_eval_realistic.json experiments/stage4_rerank_eval.json
python scripts/generate_phase4_report.py
```

报告输出：

```text
artifacts/stage4/phase4_report.md
```

## 6. 可选：跑原小评测集做对照

```bash
python evaluate_stage4.py \
  --out experiments/stage4_rerank_eval_original.json \
  --device cuda \
  --batch-size 64 \
  --reranker-device cuda \
  --reranker-batch-size 32 \
  --reranker-model-path models/bge-reranker-base
```

## 7. 打包结果下载回本地

```bash
tar -czf stage4_results.tar.gz \
  experiments/stage4_rerank_eval_realistic.json \
  experiments/stage4_rerank_eval_original.json \
  experiments/stage4_rerank_eval.json \
  artifacts/stage4/phase4_report.md
```

把 `stage4_results.tar.gz` 下载回本地后解压覆盖对应文件。

## 8. 如果只想先做轻量真实评测

```bash
python evaluate_stage4.py \
  --queries experiments/eval_queries_stage4_realistic.jsonl \
  --keyword-queries experiments/eval_queries_stage4_keyword_realistic.jsonl \
  --out experiments/stage4_rerank_eval_realistic_limit3.json \
  --device cuda \
  --batch-size 64 \
  --reranker-device cuda \
  --reranker-batch-size 32 \
  --reranker-model-path models/bge-reranker-base \
  --limit 3
```

## 9. fake 链路验证命令

fake 结果只证明代码链路可运行，不代表真实 reranker 效果：

```bash
python evaluate_stage4.py \
  --queries experiments/eval_queries_stage4_realistic.jsonl \
  --keyword-queries experiments/eval_queries_stage4_keyword_realistic.jsonl \
  --out experiments/stage4_rerank_eval_fake_realistic.json \
  --fake-reranker \
  --fake-mmr \
  --limit 2 \
  --device cuda
```
