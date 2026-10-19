# 数据资产

这里只保留数据准备 manifest，包含来源、revision、文件哈希、样本与 token 统计、划分、去重、拒收及预算策略。

训练 / 验证文本、token 数组、parquet、arrow 和模型 tokenizer 资产需要另外获取。入口为 `scripts/download_openwebmath.py`、`prepare_openwebmath_large.py`、`prepare_deepseek_coder_pilot.py`、`extend_deepseek_coder_data.py` 和 `prepare_math_sft.py`。各自使用 `--help` 查看参数；历史准备方案可从相关数据 manifest 与 `runs/*_setup/` 查到。

只有 manifest 不足以执行训练或数据依赖的集成测试。
