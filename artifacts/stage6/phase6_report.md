# 阶段六收口报告：Agent 服务化、可信引用与答案质量

## 1. 收口结论

阶段六工程闭环评测：**通过**。项目已经从检索组件升级为可通过 Python API、CLI 和 FastAPI 调用的 Agentic RAG 问答服务。

## 2. 评测边界

- 本评测验证 Agent 层的数据合同、引用映射、trace、证据不足降级和 legacy adapter；
- 采用确定性的 fake Stage5 pipeline，不使用 GPU 和外部 LLM；
- 真实召回、rerank 与 Query Planning 的效果已分别由阶段三至阶段五评测覆盖，阶段六不重复进行大规模检索测评。

## 3. 核心指标

| 指标 | 结果 |
|---|---:|
| query_count | 10 |
| answer_success_count | 8 |
| fallback_count | 2 |
| citation_valid_count | 10 |
| trace_complete_count | 10 |
| invalid_citation_detected | True |
| legacy_adapter_ok | True |
| passed_case_count | 10 |

## 4. 用例结果

| # | Query | Fallback | Citation valid | Trace complete | 结果 |
|---:|---|---:|---:|---:|---|
| 1 | 台风黄色预警下学校应该怎么做 | False | True | True | 通过 |
| 2 | 地震后高层建筑居民如何疏散 | False | True | True | 通过 |
| 3 | IV级气象灾害应急响应由谁启动 | False | True | True | 通过 |
| 4 | 暴雨预警期间地下空间要采取什么措施 | False | True | True | 通过 |
| 5 | 山洪来临时群众应该往哪里转移 | False | True | True | 通过 |
| 6 | 学校发现火灾后如何组织学生撤离 | False | True | True | 通过 |
| 7 | 极端高温时户外作业如何安排 | False | True | True | 通过 |
| 8 | 应急物资储备需要记录哪些信息 | False | True | True | 通过 |
| 9 | 知识库完全没有覆盖的量子通信问题 | True | True | True | 通过 |
| 10 | 一个只有微弱相关证据的问题 | True | True | True | 通过 |

## 5. 阶段六产物

- `src/rag_v2/agent/models.py`：Evidence、Citation、Trace、RagAnswer 数据合同；
- `src/rag_v2/agent/service.py`：封装 Stage5 的统一问答服务；
- `src/rag_v2/agent/evidence.py`、`composer.py`：证据编号与可引用模板回答；
- `src/rag_v2/agent/verifier.py`、`guardrail.py`：引用校验与证据不足降级；
- `src/rag_v2/agent/legacy_adapter.py`、`api.py`：旧接口适配与 FastAPI；
- `scripts/answer_stage6.py`：CLI 演示；
- `evaluate_stage6.py`：阶段六确定性工程评测。

## 6. 简历/面试表述

> 将前五阶段的多路检索链路封装为 Agentic RAG 问答服务，设计 Evidence/Citation/RagAnswer 数据合同，实现证据编号、可追溯引用、确定性引用校验与证据不足降级，并通过 legacy adapter、CLI 和 FastAPI 提供兼容旧 Agent 的服务化接口；使用 trace 暴露 Query Plan、检索分支和证据数量，完成可解释、可降级的端到端闭环。
