# newrag

独立 RAG 优化项目，原项目路径：

``text
G:\tiaozhanbei\Youan-AI-main\youan-multiagent\multi_agent_server\app\RAG
``

阶段 0 目标：隔离原代码、冻结 legacy 行为、建立轻量可复现 baseline。

## 目录

- `legacy_snapshot/RAG/`：原 RAG 代码和数据快照，不包含 `bge-large-zh-v1.5` 大模型目录。
- `src/`：后续新 RAG 实现目录。
- `experiments/eval_queries.jsonl`：轻量评测集。
- `evaluate.py`：legacy baseline 评测脚本。
- `docs/`：阶段记录。

## 模型路径

默认复用原项目模型目录：

``text
G:\tiaozhanbei\Youan-AI-main\youan-multiagent\multi_agent_server\app\RAG\bge-large-zh-v1.5
``
