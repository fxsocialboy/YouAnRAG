# Stage 7.4 After-fix AutoDL运行手册

本阶段本地开发不需要GPU。上传代码包并合并已有的两个模型目录后，先运行小规模校准：

```bash
cd /root/autodl-tmp/YouAnRAG
bash scripts/run_stage74_calibration_autodl.sh
```

只有输出以下内容才能继续全量实验：

```text
[OK] Stage7.4 calibration passed
```

随后运行正式评测：

```bash
cd /root/autodl-tmp/YouAnRAG
bash scripts/run_stage7_autodl.sh
```

脚本会依次完成Qdrant恢复、CUDA/模型/索引/数据集/DeepSeek预检、三组正式评测、
before/after报告、Legacy hash诊断和结果打包。断点续跑仅会复用`config_hash`一致且
状态为`ok`的样本；参数或数据发生变化时会拒绝混用旧结果。

成功标志：

```text
[OK] Stage7.4 after-fix evaluation finished
```

下载文件：

```text
/root/autodl-tmp/YouAnRAG/artifacts/stage74_after_fix_results.tar.gz
```

所有新结果写入`artifacts/stage7/after_fix/`，不会覆盖`before_fix`基线。
