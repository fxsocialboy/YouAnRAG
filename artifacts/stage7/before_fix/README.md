# Stage 7.4 before-fix 基线

本目录保存阶段7.4修复前的三次AutoDL正式评测结果、汇总报告和运行日志。文件内容由
`before_fix_manifest.json`中的SHA-256固定，不应在后续修复中覆盖。

在项目根目录执行一条命令即可校验：

```bash
python scripts/archive_stage74_before_fix.py --verify-only
```

失败回归集位于`experiments/stage74_fix_regression.jsonl`，其来源和数量记录在
`stage74_regression_manifest.json`中。原25条标注集与120条随机集保持冻结。
