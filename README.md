# Chicago TNP Analysis Notebooks / Chicago TNP 分析 Notebook

The notebooks are ordered by experiment dependency and project progression.  
Notebook 按实验依赖和项目推进顺序排列。

1. `01_hourly_h3_demand_forecasting_and_spatial_analysis.ipynb`  
   Hourly anomaly sensitivity, multi-horizon forecasting, pickup/dropoff flow, and H3 spread. / 小时异常敏感度、多步预测、上下车流动和 H3 扩散。
2. `02_two_stage_tip_amount_prediction.ipynb`  
   Two-stage expected recorded-tip prediction. / 两阶段记录小费期望值预测。
3. `03_llm_peak_continuation_benchmark.ipynb`  
   OpenAI, GBM, and deterministic baselines for peak continuation. / 大模型、GBM 与确定性高峰持续基准。
4. `04_recorded_tip_llm_vs_tree_comparison.ipynb`  
   Recorded-tip yes/no comparison between LLM and tree models. / 大模型与树模型的记录小费二分类对比。
5. `05_chronos2_vs_lightgbm_forecasting_benchmark.ipynb`  
   Chronos-2 and LightGBM hourly forecast benchmark. / Chronos-2 与 LightGBM 小时预测基准。

## Configuration / 配置

```bash
export CHICAGO_TNP_PROJECT_DIR="/path/to/chicago_tnp_full"
export MATRIXONE_HOST="127.0.0.1"
export MATRIXONE_PORT="6001"
export MATRIXONE_USER="root"
export MATRIXONE_DATABASE="chicago_tnp"
export MATRIXONE_PASSWORD="your-password"
```

OpenAI API keys are requested interactively in the notebooks that use an LLM. They are not stored in source or output files.  
使用大模型的 Notebook 会在运行时交互式读取 OpenAI API key，不会将其写入源码或输出文件。

Python functions use standard English `snake_case` identifiers for compatibility. Every function includes an English/Chinese docstring.  
Python 函数使用兼容工具链的英文 `snake_case` 标识符，每个函数均提供中英双语 docstring。
