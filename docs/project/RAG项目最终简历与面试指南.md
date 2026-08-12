# RAG项目最终简历与面试指南

## 一、推荐项目名称

**YouAnRAG：面向灾害应急多智能体系统的混合检索与可信生成系统**

## 二、简历项目描述

面向 51 份灾害应急预案构建可接入原 LangGraph 多智能体系统的完整 RAG V2，解决原实现仅依赖 FAISS 单路向量匹配、分块粗糙、索引需手工重建且缺乏精排与答案质量控制的问题。

### 推荐简历 bullet

1. 设计 Markdown 结构感知分块和增量索引链路，使用 **BGE + Qdrant Local + SQLite** 实现文档新增、修改、删除检测与向量库自动同步，替代原有手工更新 FAISS 的方式。
2. 构建 **Qdrant 语义召回 + BM25 关键词召回 + RRF 融合 + Cross-Encoder 精排 + MMR 去冗余** 的多阶段检索链路，并通过上下文预算组装保留章节路径与相邻语义信息。
3. 引入 **Query Rewrite、Multi-Query 与条件 HyDE**，根据查询类型动态选择扩展策略；在 25 条冻结标注集上将 Chunk Recall@10 从 **58.70% 提升至 76.09%**，Chunk MRR@10 从 **36.86% 提升至 46.39%**。
4. 实现 DeepSeek 引用式生成、引用映射校验、Evidence Guardrail 和证据不足降级；120 条固定种子随机测试中 Faithfulness 达 **97.02%**、Answer Relevancy 达 **94.62%**、OOD 拒答准确率 **100%**、未知引用数为 **0**。
5. 通过 Adapter 将 RAG V2 接入原 `DisasterResponseAgent` / LangGraph，并保留 Legacy 后端回退；完成 RTX 4090 上的真实 Embedding、Rerank、DeepSeek 端到端评测与 3 个真实 Agent 场景烟测。

简历空间有限时保留第 2、3、4、5 条。

## 三、30秒面试介绍

> 原项目的 RAG 只有 FAISS 向量相似度匹配，文档变化后需要手工重建，而且没有关键词召回、精排、查询改写和答案可信度控制。我把它独立重构为 RAG V2：离线侧做结构感知分块，用 SQLite 记录文件状态、Qdrant 保存向量并支持增量同步；在线侧用 Qdrant 和 BM25 双路粗召回，经 RRF 融合、Cross-Encoder 精排、MMR 去冗余后组装上下文，再根据 Query Analyzer 选择 Rewrite、Multi-Query 或 HyDE，最后由 DeepSeek 生成带引用答案并经过 Guardrail 校验。新链路已通过 Adapter 接入原 LangGraph，Chunk Recall@10 从 58.7% 提升到 76.09%，随机集 Faithfulness 为 97.02%。

## 四、重点追问

### 为什么同时使用Qdrant和SQLite？

Qdrant负责高维向量存储和相似度检索；SQLite负责文档哈希、更新时间、chunk映射和同步状态。二者职责不同：前者是检索引擎，后者是轻量元数据与增量更新控制面。

### 为什么需要BM25？

向量召回擅长语义近似，但对政策编号、响应等级、机构名称和专有词不一定稳定。BM25根据词频和逆文档频率召回精确关键词，再用RRF与稠密结果融合，可以互补而不要求两路分数同尺度。

### 粗排和精排如何区分？

粗排由Qdrant和BM25分别快速召回较多候选并通过RRF融合；精排用Cross-Encoder同时编码query与候选文本，计算更准确的相关性。Cross-Encoder计算量较大，只处理粗排后的少量候选。

### MMR解决什么问题？

精排Top结果可能来自相同章节、内容高度重复。MMR在相关性和候选间多样性之间权衡，避免上下文预算被重复chunk占满。

### Rewrite、Multi-Query和HyDE是否全部无条件执行？

不是。Query Analyzer根据短查询、歧义、场景型或复杂问题选择策略。Rewrite负责消歧和规范化，Multi-Query从多个表达角度召回，HyDE只在复杂语义查询且普通召回可能不足时启用，避免无谓增加延迟和噪声。

### 如何避免模型幻觉？

上下文被转换为带稳定编号的Evidence，Composer必须按编号引用；生成后校验引用是否存在、关键陈述是否有证据，并根据证据充分性决定回答还是Fallback。未知引用不会直接返回给用户。

### Faithfulness是怎样评测的？

将答案拆成可验证事实，让固定Prompt、温度0的Judge逐条判断能否由检索证据支持，再聚合为Faithfulness。由于Composer和Judge都使用DeepSeek，报告明确披露同模型偏差，因此同时保留确定性的引用映射和Fallback指标，并冻结逐条判断理由供抽查。

### 为什么V2延迟更高？

Legacy只做一次向量搜索；V2增加了查询分析、可能的LLM改写、双路召回、Cross-Encoder和答案生成。当前目标是验证完整链路和效果。生产化可缓存Rewrite与Embedding、批量Rerank、将模型服务化，并按查询复杂度跳过HyDE或精排。

### 为什么随机领域内回答率只有78%？

知识库只有51份文档，随机问题不一定能被现有文档充分支持。系统选择在证据不足时保守降级；该指标表示回答覆盖率而非准确率。继续放宽阈值虽然能提升覆盖率，但可能损害Faithfulness和OOD安全性。

## 五、指标表达边界

- 可以说“25条冻结标注集上的Recall/MRR/nDCG”；
- 可以说“120条固定种子随机鲁棒性测试上的Faithfulness和引用指标”；
- 不要把120条无相关性标签随机集说成准确率测试；
- 不要声称是大规模并发或生产部署；
- Qdrant Local可表述为“使用Qdrant向量数据库的本地持久化模式”，不要虚构Qdrant Server集群经历。
