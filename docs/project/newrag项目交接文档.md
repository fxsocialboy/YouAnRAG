# newrag 项目交接文档 — 给新 Codex Agent 快速上手

## 1. 项目背景

这是"佑安"灾害救援多智能体项目的 RAG 模块独立优化工程。原 RAG 代码位于 `G:\tiaozhanbei\Youan-AI-main\`，存在切分、编码、检索等多处缺陷。优化工作在隔离目录 

```
G:\tiaozhanbei\newrag\
```

进行，**不修改原始项目代码**。

### 核心约束

- **用户是 CS 研二学生，秋招简历项目，不是生产系统**
- **用户不写代码，只读代码和文档，由你（Codex）完成所有开发**
- 每个阶段完成后必须写测试，测试必须全部通过
- 优先完成对面试最有价值的阶段
- 不要在 CPU 机器上运行全量 BGE embedding（6269 条文本 CPU 编码需 15~20 分钟）；GPU 或已有向量直接复用
- 所有产物写入 `artifacts/` 目录，不污染源码

---

## 2. 技术架构总览

### 存储体系

| 存储 | 路径 | 用途 |
|---|---|---|
| Stage1 FAISS 索引 | `artifacts/stage1/faiss_index.index` | 6269 个 BGE 1024 维向量，已 gitignore |
| Stage1 元数据 | `artifacts/stage1/chunk_metadata.json` | 6269 条 chunk 记录，所有后续阶段的输入源 |
| Qdrant Local | `artifacts/qdrant_local/` | 向量数据库，collection `youan_rag_stage2`，6269 points |
| SQLite Registry | `artifacts/stage2/document_registry.sqlite3` | documents + chunk_mappings 两张表 |
| Stage3 BM25 索引 | `artifacts/stage3/bm25_index.json` | 6269 文档的 BM25 词频索引，2.1MB |

### 检索管线（当前最新）

```
query
├─ Qdrant dense  → Top-30 (BGE 1024维，COSINE)
├─ BM25 sparse   → Top-30 (手写，无第三方依赖)
│     └─ RRF fusion (k=60)
├─ Metadata filter (仅 source_file 硬过滤 + Reserved Hooks)
└─ Context packer (邻 chunk 合并 + 同 section 补全 + token budget)
      └─ [S1] / [S2] citation evidence
```

### 代码目录结构

```
newrag/
├── src/rag_v2/
│   ├── config.py           # 全局配置（所有路径、参数）
│   ├── schemas.py          # 数据模型定义
│   ├── ingestion/
│   │   ├── normalizer.py    # Markdown 清洗
│   │   ├── markdown_parser.py # Markdown → MarkdownBlock
│   │   ├── token_counter.py  # 轻量 token 计数器
│   │   ├── chunker.py       # Token-budget 分块器（8 规则）
│   │   └── metadata.py      # 质量报告 + metadata 丰富
│   ├── embedding/
│   │   └── bge_embedder.py   # BGE 封装（instruction 方向已修正）
│   ├── stores/
│   │   ├── vector_store.py   # VectorStore Protocol（7 方法接口）
│   │   ├── faiss_store.py    # FAISS 实现（Stage1 用）
│   │   └── qdrant_store.py   # Qdrant 实现（uuid5 幂等，COSINE）
│   ├── sync/
│   │   ├── registry.py       # SQLite Registry（documents + chunk_mappings）
│   │   └── file_tracker.py   # 文件 hash 扫描 + 变更检测
│   └── retrieval/
│       ├── qdrant_searcher.py    # Qdrant 检索封装
│       ├── bm25_index.py         # 手写 BM25（246行，无第三方依赖）
│       ├── hybrid_searcher.py    # RRF 融合（Dense + Sparse）
│       ├── filters.py            # Metadata filter（source_file + Reserved Hooks）
│       ├── context_packer.py     # 上下文组装（邻 chunk 合并 + evidence）
│       └── stage1_searcher.py    # Stage1 旧版兼容
├── scripts/                 # CLI 构建/运行脚本
├── tests/                   # pytest 测试（当前 ~100+ 全部通过）
├── experiments/             # 评测集和结果
│   ├── eval_queries_stage0.jsonl     # 25 条自然语言 query
│   └── eval_queries_keyword.jsonl    # 10 条关键词/条款型 query
├── artifacts/stage1/        # 6269 chunks + metadata
├── artifacts/stage2/        # Qdrant 导入报告 + Registry
├── artifacts/stage3/        # BM25 索引 + 评测结果
└── .gitignore               # 已排除 qdrant_local/、pytest_tmp/、archive/
```

---

## 3. 已完成阶段（0~3）

### 阶段 0：评测基线

- 25 条自然语言 query 集
- 原项目 RAG vs 阶段一 RAG 的 baseline 对比评测脚本
- 状态：✅ 收口

### 阶段 1：数据管道重构

- Markdown 清洗（保留英文空格、HTML unescape、NFC 归一化）
- Token-budget chunker（280/360/448 三阈值 + overlap=50）
- BGE instruction 方向修正（passage 不加 instruction，query 加）
- 6266 chunks → 6269 vectors（FAISS IndexFlatL2）
- 状态：✅ 收口

### 阶段 2：Qdrant 向量数据库 + SQLite 增量同步

- VectorStore Protocol + QdrantStore 实现（uuid5 幂等 + COSINE）
- FAISS reconstruct() 导入 Qdrant（6269 points，零误差）
- SQLite Registry（documents + chunk_mappings 两张表）
- FileTracker（SHA256 文档 hash → 变更检测）
- sync_qdrant_stage2.py 同步脚本（--scan-only / --sync）
- 检索一致性验证：25 query Top-10 = 100% 匹配（含顺序）
- 状态：✅ 收口

### 阶段 3：混合召回与上下文组装

- 手写 BM25（k1=1.5, b=0.75，n-gram 中文分词，JSON 序列化）
- RRF 融合 Qdrant dense + BM25 sparse（k=60）
- Metadata filter（source_file 硬过滤 + region/hazard/type Reserved Hooks）
- Context packer（邻 chunk 合并 + 同 section 补充 + token budget=1200）
- 评测：自然语言 query dense 0.96→hybrid 1.00；关键词 query dense 0.90→hybrid 1.00
- 状态：✅ 收口

---

## 4. 待开发阶段

### 阶段 4：Cross-encoder 精排 + MMR 多样性控制 (P1)

**目标**：在 hybrid RRF 粗排（Top-30）之后，加一层 Cross-encoder 精排和 MMR 去重。

**核心任务**：
- 集成 Cross-encoder reranker（如 `BAAI/bge-reranker-large`），pair-wise 重打分
- 实现 MMR 去重（Maximal Marginal Relevance），Lambda 默认 0.7
- 输出 `dense_score`、`sparse_score`、`fusion_score`、`rerank_score`、`mmr_score` 全链路
- 评测 rerank + MMR vs 当前 hybrid 的效果差异

**约束**：
- Cross-encoder 模型较大（~1.3GB），首次加载需要下载
- 精排 Top-30 → Top-10，不改变前面阶段的召回逻辑
- 需保持 `invoke()` 兼容

### 阶段 5：Query Rewrite / Multi-Query / HyDE (P1)

**目标**：在检索前增强 query 质量。

**核心任务**：
- Multi-Query：一个用户问题生成 2~3 个等价改写 query，分别检索后合并
- HyDE：用 LLM 先生成假设答案 → 对假设答案做 dense 召回 → 和原 query 结果合并（作为可选分支）
- 评测改写/HyDE 对召回的提升

**约束**：
- HyDE 需要调用 LLM API（可接智谱 GLM-4），是网络调用
- 用户可能没有 API key，优先实现 Multi-Query（纯本地改写逻辑）
- HyDE 标记为 optional，不影响默认检索流程

### 阶段 6：Agent 集成、可信引用与答案质量 (P0/P1)

**目标**：让 RAG 输出能被下游 Agent/LLM 直接消费。

**核心任务**：
- 对接原项目 `retrieve_node` 的调用接口（`invoke() → List[str]`）
- 输出结构化 evidence（带 `citation_id` [S1]/[S2]、`source_file`、`section_path`）
- Evidence 格式化为 XML/JSON 注入 prompt
- 离线评测 retrieval@k、faithfulness、citation correctness
- Legacy adapter：保持与原项目 `RAG.invoke()` 兼容

**约束**：
- 不修改原项目代码（`G:\tiaozhanbei\Youan-AI-main\`）
- 只需要提供兼容接口即可，不需要实际部署到原项目

### 阶段 7：评测、观测与生产化 (P2)

**目标**：工程收口，为项目画句号。

**核心任务**：
- 全链路回归测试
- Redis query cache（embedding + 检索结果）
- 日志与 metrics
- 最终评测报告

**约束**：
- 可选阶段，简历项目不需要全部完成
- 优先完成阶段 4、6 的最小闭环

---

## 5. 关键决策记录（不要推翻）

| 决策 | 结论 |
|---|---|
| 向量数据库 | Qdrant Local Mode，不装 Docker，不加新服务 |
| BM25 实现 | 纯手写 246 行，无 jieba/elasticsearch/rank-bm25 依赖 |
| 文档同步 | 手动脚本触发，不用常驻服务 |
| Metadata filter | 第一版只做 source_file，不盲目激活 region/hazard |
| 上下文组装 | 内部索引找邻居（不从 Qdrant 再查），section 边界检查 |
| Token 定价 | token=字符/2（中文），不含 tiktoken 依赖 |
| 评测 query | 25 条自然语言 + 10 条关键词，全部 JSONL |
| Agent 接口 | 保留 invoke() 兼容，额外提供 search() 结构化输出 |

---

## 6. 快速上手

### 运行测试

```powershell
cd G:\tiaozhanbei\newrag
python -m pytest tests/ -v
```

当前全量 ~100+ 测试，全部通过。开发新功能时确保**全量回归通过**。

### 关键路径

- 核心配置：`src/rag_v2/config.py` 的 `RagV2Config`
- 所有 artifacts 输入源：`artifacts/stage1/chunk_metadata.json`（6269 条）
- BM25 索引构建：`python scripts/build_bm25_stage3.py`
- Qdrant 数据导入：`python scripts/import_stage1_to_qdrant.py`
- 文档同步：`python scripts/sync_qdrant_stage2.py --scan-only`

### Python 环境

```
Python 3.14.4
faiss-cpu         1.15.0
pytest            8.4.2
pytest-asyncio    0.26.0
qdrant-client     1.18.0
sentence-transformers  5.6.0
torch             2.12.0
transformers      5.12.1
```

没有 `requirements.txt`，上述是当前环境的实际版本。安装新包时用 pip，但**不要引入 jieba、rank-bm25、elasticsearch**（BM25 已手写）。

### 重要：不要重新跑全量 embedding

Stage1 的 6269 个 BGE 向量已由上一任在 GPU 上生成（69 秒），CPU 重跑需要 15~20 分钟。新开发应复用已有向量（FAISS reconstruct 或 Qdrant 直接读）。

### GitHub

```
Remote: https://github.com/fxsocialboy/YouAnRAG.git
Branch: main
```

用户自主管理推送，你不需要执行 git push。

### 测试：104 条

```powershell
cd G:\tiaozhanbei\newrag
python -m pytest tests/ -v    # 104 passed
```

**规则：任何代码变更后，全量回归必须保持 104/104 通过。** 新功能增加新测试文件，`test_<模块名>.py` 命名规范。

### 常见踩坑

| 坑 | 说明 |
|---|---|
| `config.py` 是 frozen dataclass | 字段创建后不可改，需要改配置就改 `default()` 里的路径 |
| BGE 模型路径 | 硬编码在 `config.py` 第 62 行，指向原项目目录下的 `bge-large-zh-v1.5`（~1.3GB） |
| 全量 BGE 编码 | 6269 条文本 CPU 编码需 15~20 分钟，不要重跑。仅少量新 chunk 可以在 CPU 跑 |
| Qdrant 幂等 | point id 由 `uuid.uuid5(chunk_id)` 生成，重复 upsert 不会产生重复数据 |
| BM25 索引重建 | 仅需 `python scripts/build_bm25_stage3.py`，1.88 秒完成 |
| 所有路径用 Path | 不要硬编码 Windows 路径，用 `config.py` 或 `Path(__file__)` 相对解析 |

---

## 7. 下一步

请从 **阶段 4**（Cross-encoder 精排 + MMR）开始开发。先阅读：

1. `G:\tiaozhanbei\aidoc\RAG优化\RAG完整分阶段优化方案.md` — 完整 7 阶段方案
2. `G:\tiaozhanbei\aidoc\RAG优化\阶段三分阶段开发计划.md` — 阶段三详细设计（已完成，参考架构）
3. `G:\tiaozhanbei\newrag\src\rag_v2\retrieval\hybrid_searcher.py` — 当前 hybrid searcher 输出结构
4. `G:\tiaozhanbei\newrag\src\rag_v2\config.py` — 全局配置

生成阶段四分阶段开发计划后，先让用户审核再开始写代码。
