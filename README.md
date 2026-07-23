# Chicago TNP Analysis and Model Deployment
# Chicago TNP 数据分析与模型部署

This repository explores Chicago Transportation Network Provider (TNP) trip data from 2022 to 2024. The project uses MatrixOne for full-data aggregation, H3 cells for spatial analysis, and Python models for demand, peak, and recorded-tip prediction.
本仓库分析 2022 至 2024 年的 Chicago Transportation Network Provider（TNP）行程数据。项目使用 MatrixOne 完成全量数据聚合，使用 H3 网格进行空间分析，并通过 Python 模型预测客流、高峰持续情况和记录小费。

The notebooks are numbered in the order the experiments were developed. Each notebook can also be read as a separate experiment.
Notebook 按实验推进顺序编号，同时每个 Notebook 也可以作为一个独立实验阅读。

## Notebook overview / Notebook 内容概览

### 01. Hourly H3 demand forecasting and spatial analysis
### 01. 小时级 H3 客流预测与空间分析

`01_hourly_h3_demand_forecasting_and_spatial_analysis.ipynb`

This notebook starts with hourly pickup demand in each H3 cell. It checks how sensitive the anomaly score is, predicts demand 1, 2, 3, and 6 hours ahead, compares pickup and dropoff flows, and looks at whether a peak spreads to nearby H3 cells. In simple terms, it asks: when one area becomes busy, what may happen next in the same area and around it?
这个 Notebook 从每个 H3 网格的小时上车量开始，检查异常分数的敏感度，预测未来 1、2、3、6 小时的客流，对比上车和下车流量，并分析高峰是否会扩散到附近的 H3 网格。简单来说，它研究的是：一个区域开始变忙之后，这个区域和周边区域接下来可能发生什么。

### 02. Two-stage recorded-tip amount prediction
### 02. 两阶段记录小费金额预测

`02_two_stage_tip_amount_prediction.ipynb`

This notebook predicts the expected recorded tip for a completed trip. It first predicts whether the trip has a positive recorded tip, and then predicts the tip amount only for trips with a positive tip. The two results are combined into one expected-tip value. It also compares different feature groups to show how pickup time, dropoff time, location changes, fare, distance, and trip duration help the prediction.
这个 Notebook 预测一笔已完成行程的记录小费期望值。模型先判断该行程是否记录了正小费，再只针对有正小费的行程预测小费金额，最后把两个结果合并成一个期望小费值。实验还比较了不同特征组，用来说明上下车时间、地点变化、车费、距离和行程时长对预测有什么帮助。

### 03. LLM peak-continuation benchmark
### 03. 大模型高峰持续性预测对比

`03_llm_peak_continuation_benchmark.ipynb`

This notebook focuses on one direct question: after an H3 cell is already in a peak, will the peak still be active 1, 3, or 6 hours later? It gives the same compact hourly summary to deterministic rules, a gradient-boosting model, and an OpenAI model, then compares their results on the same 2024 cases. The LLM reads aggregated features instead of the full trip table.
这个 Notebook 专注于一个直接的问题：当一个 H3 网格已经处于高峰时，1、3、6 小时后高峰是否仍会持续。实验把相同的小时级摘要分别交给确定性规则、梯度提升模型和 OpenAI 模型，并在相同的 2024 年案例上比较结果。大模型读取的是聚合后的特征，而不是完整行程表。

### 04. Recorded-tip LLM vs. tree comparison
### 04. 记录小费的大模型与树模型对比

`04_recorded_tip_llm_vs_tree_comparison.ipynb`

This notebook compares several ways to answer a yes-or-no question: does this trip have a positive recorded tip? A shallow decision tree shows a small set of readable rules, LightGBM provides a stronger machine-learning baseline, and an OpenAI model makes the same decision from a structured trip summary. This makes it possible to compare predictive quality, model cost, and how easy each method is to explain.
这个 Notebook 比较几种方法对同一个二分类问题的判断：这笔行程是否记录了正小费。浅层决策树展示少量容易阅读的判断规则，LightGBM 提供更强的机器学习基准，OpenAI 模型则根据结构化行程摘要作出同样的判断。这样可以同时比较预测效果、模型成本和解释难度。

### 05. Chronos-2 vs. LightGBM forecasting benchmark
### 05. Chronos-2 与 LightGBM 客流预测对比

`05_chronos2_vs_lightgbm_forecasting_benchmark.ipynb`

This notebook compares time-series and feature-based forecasting on the same hourly H3 task. It uses the previous 28 days to predict the next 24 hours and compares a last-week baseline, several Chronos-2 settings, and LightGBM models. Every method uses the same forecast times and targets, so the comparison stays fair.
这个 Notebook 在同一个小时级 H3 客流任务上比较时间序列模型和特征模型。实验使用过去 28 天预测未来 24 小时，并对比上周同期基线、多种 Chronos-2 设置和 LightGBM 模型。所有方法使用相同的预测时间点和目标，因此结果可以公平比较。

## ONNX and Go model deployments / ONNX 与 Go 模型部署

### H3 next-hour demand model / H3 下一小时客流模型

`model_h3_next_hour_onnx_go_deployment/`

This folder turns one forecasting experiment into a small deployment package. The model is trained in Python, exported to ONNX, and loaded in Go to predict completed pickup trips in the same H3 cell one hour later. It includes the ONNX model, the fixed 61-feature schema, Python validation code, Go inference code, test cases, and a macOS Apple Silicon ONNX Runtime library.
这个文件夹把其中一个客流预测实验整理成可部署的小型项目。模型在 Python 中训练并导出为 ONNX，然后由 Go 加载，用来预测同一 H3 网格下一小时的已完成上车订单数。文件夹包含 ONNX 模型、固定的 61 个特征结构、Python 验证代码、Go 推理代码、测试样本，以及适用于 macOS Apple Silicon 的 ONNX Runtime 运行库。

Run the included parity test from the repository root:
在仓库根目录运行下面的跨语言一致性测试：

```bash
./model_h3_next_hour_onnx_go_deployment/run_go_test.sh
```

A final `PASS` means the Go program and Python ONNX Runtime produce matching predictions within the configured tolerance. See the folder's own `README.md` for setup, inference, retraining, and new-machine reproduction steps.
最后显示 `PASS`，表示 Go 程序与 Python ONNX Runtime 的预测结果在设定误差范围内一致。环境配置、模型推理、重新训练和新机器复现步骤请查看该文件夹内的 `README.md`。

### Two-stage recorded-tip model / 两阶段记录小费模型

`model_two_stage_tip_onnx_go_deployment/`

This folder deploys the two-stage recorded-tip model used in the tip prediction experiment. The first ONNX model estimates whether a positive tip is recorded, and the second ONNX model estimates the amount when a positive tip is recorded. The Go program combines them as `tip probability × positive-tip amount` to produce the expected recorded tip. The package includes both ONNX files, a fixed 17-feature schema, Python and Go validation code, example inputs, model metrics, and bilingual documentation.
这个文件夹部署小费预测实验中的两阶段记录小费模型。第一个 ONNX 模型判断是否记录正小费，第二个 ONNX 模型预测记录正小费时的金额。Go 程序使用“记录小费概率 × 正小费金额”计算记录小费期望值。部署包包含两个 ONNX 文件、固定的 17 个特征结构、Python 与 Go 验证代码、示例输入、模型指标和中英双语文档。

Run its parity test from the repository root:
在仓库根目录运行它的跨语言一致性测试：

```bash
./model_two_stage_tip_onnx_go_deployment/run_go_test.sh
```

A final `PASS` confirms that both ONNX stages produce matching results in Python and Go. The model improves expected-tip MAE over a simple training-mean baseline, but it remains a baseline because tipping behavior also depends on information that is not available in the trip table.
最后显示 `PASS`，说明两个 ONNX 阶段在 Python 和 Go 中得到一致结果。该模型的期望小费 MAE 优于简单训练期平均值基线，但由于小费行为还受到行程表中没有的信息影响，因此它仍然是一个基线模型。

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
