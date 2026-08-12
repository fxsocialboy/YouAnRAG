# Stage7 最终评测集变更记录

## 冻结规则

- 首次冻结日期：2026-08-11；
- `eval_queries_final_labeled.jsonl`：25 条人工标注质量集；
- `eval_queries_final_random.jsonl`：120 条固定种子（`20260811`）随机鲁棒性集；
- 冻结后不得为了提高指标增删 query、调整分组或修改 `expected_fallback`；
- 允许修复客观错误，例如文件名、chunk_id 或事实标注错误；
- 每次修复必须在下表记录日期、query_id、修改内容、原因和提交标识。

## 变更记录

| 日期 | Query ID | 修改内容 | 原因 | Commit |
|---|---|---|---|---|
| 2026-08-11 | ALL | 首次创建并冻结双评测集 | 阶段 7.1 基线 | 待提交 |
| 2026-08-11 | ALL | 增加显式 `disaster_type` 字段 | 支持最终报告按灾种分组；避免运行时从文件名推断 | 待提交 |

## 指标边界

标注质量集可以计算 Recall、MRR、nDCG 和 fallback accuracy。随机鲁棒性集没有人工相关性标签，因此不计算 Recall/MRR，只用于回答率、fallback、引用映射、LLM-as-Judge、错误率和延迟分析。
