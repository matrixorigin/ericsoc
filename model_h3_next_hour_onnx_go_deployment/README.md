# Chicago TNP H3 Next-Hour Demand: ONNX and Go Deployment  
# Chicago TNP H3 下一小时客流预测：ONNX 与 Go 部署

## Project overview / 项目概览

This project trains a model in Python, exports it to ONNX, and runs the same model in Go. Given 61 features available at the end of the current hour, the model predicts completed pickup trips in the same H3 cell during the next hour.  
本项目在 Python 中训练模型，将模型导出为 ONNX，并在 Go 中运行同一个模型。输入当前小时结束时已经获得的 61 个特征，模型预测同一 H3 单元下一小时的已完成上车订单数。

```text
MatrixOne hourly H3 aggregates
        ↓
Python feature engineering and model training
        ↓
ONNX model + fixed feature schema
        ↓
Python ONNX Runtime and Go ONNX Runtime
        ↓
Cross-language parity validation
```

```text
MatrixOne 小时级 H3 聚合
        ↓
Python 特征工程与模型训练
        ↓
ONNX 模型与固定特征结构
        ↓
Python ONNX Runtime 与 Go ONNX Runtime
        ↓
跨语言一致性验证
```

## Repository contents / 目录内容

| Path / 路径 | Purpose / 用途 |
|---|---|
| `06_h3_next_hour_onnx_training_and_go_validation.ipynb` | Reproducible training, export, and validation workflow. / 可复现的训练、导出与验证流程。 |
| `python/feature_engineering.py` | Builds leakage-safe hourly features. / 构造避免未来信息泄漏的小时级特征。 |
| `python/train_export.py` | Trains the model and exports ONNX artifacts. / 训练模型并导出 ONNX 产物。 |
| `python/verify_onnx.py` | Verifies Python ONNX predictions. / 验证 Python ONNX 预测。 |
| `go/main.go` | Loads the ONNX model and performs Go inference. / 加载 ONNX 模型并执行 Go 推理。 |
| `model/h3_next_hour_pickups.onnx` | Cross-language model file. / 跨语言模型文件。 |
| `model/feature_schema.json` | Ordered feature contract, medians, and H3 codes. / 固定特征顺序、中位数和 H3 编码。 |
| `model/model_metrics.json` | Test metrics and Python-to-ONNX parity results. / 测试指标与 Python-ONNX 一致性结果。 |
| `testdata/parity_test_cases.json` | Twenty fixed cross-language validation cases. / 20 条固定跨语言验证样本。 |
| `testdata/sample_prediction_input.json` | Example input without reference answers. / 不含参考答案的预测输入示例。 |
| `runtime/macos-arm64/libonnxruntime.dylib` | ONNX Runtime library for the included Apple Silicon demonstration. / 随包提供的 Apple Silicon 演示运行库。 |

## Model and result summary / 模型与结果概览

- Model: `HistGradientBoostingRegressor` with absolute-error loss. / 模型：使用绝对误差损失的 `HistGradientBoostingRegressor`。
- Training period: 2022-2023; test period: 2024. / 训练期：2022-2023；测试期：2024。
- Training rows: 875,750 H3-hours; test rows: 439,100 H3-hours. / 训练样本：875,750 个 H3-hour；测试样本：439,100 个 H3-hour。
- ONNX test result: MAE 12.66, RMSE 24.39, R² 0.9412. / ONNX 测试结果：MAE 12.66、RMSE 24.39、R² 0.9412。
- Go/Python validation: maximum difference across 20 fixed cases is 0.000183 pickups. / Go/Python 验证：20 条固定样本最大差异为 0.000183 个 pickup。

The target is completed pickup trips, not every ride request. The model estimates demand observed in completed-trip data and does not directly measure cancellations, wait time, or available drivers.  
预测目标是已完成上车订单数，不是全部叫车请求。模型估计完成订单数据中观测到的客流，不直接测量取消、等待时间或可用司机数。

## Environment / 环境配置

### Required for Go inference / Go 推理必需环境

- Go 1.24 or newer. / Go 1.24 或更高版本。
- ONNX Runtime shared library. / ONNX Runtime 动态库。
- The included runtime supports macOS Apple Silicon. / 随包运行库支持 macOS Apple Silicon。

For another operating system, install a matching ONNX Runtime shared library and set:  
其他操作系统需要安装对应的 ONNX Runtime 动态库，并设置：

```bash
export ONNXRUNTIME_SHARED_LIBRARY="/absolute/path/to/libonnxruntime.so"
```

Use `.dylib` on macOS, `.so` on Linux, and `.dll` on Windows.  
macOS 使用 `.dylib`，Linux 使用 `.so`，Windows 使用 `.dll`。

### Required for Python validation or retraining / Python 验证或重新训练所需环境

- Python 3.12 is the validated environment. / 已验证环境为 Python 3.12。
- Install the locked dependencies in an isolated environment. / 在独立环境中安装固定版本依赖。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MatrixOne is not required for model inference. It is used upstream to create the complete hourly H3 cache needed for retraining.  
模型推理不需要 MatrixOne。MatrixOne 仅在上游生成重新训练所需的完整小时级 H3 缓存。

## Quick demonstration / 快速演示

Run all commands from this directory.  
所有命令均在本目录执行。

### 1. Go and Python parity test / Go 与 Python 一致性测试

```bash
./run_go_test.sh
```

The command compiles the Go source, loads the included ONNX model, predicts the same 20 cases used by Python ONNX Runtime, and checks a tolerance of 0.001 pickups. A successful run ends with:  
该命令会编译 Go 源码、加载随包 ONNX 模型、预测 Python ONNX Runtime 使用的相同 20 条样本，并检查 0.001 个 pickup 的误差容限。成功时最后显示：

```text
PASS: Go and Python ONNX Runtime predictions match within tolerance.
```

This result demonstrates that the Python-trained model produces operationally identical predictions in Go.  
该结果说明 Python 训练的模型在 Go 中能够得到业务上相同的预测。

### 2. Python ONNX validation / Python ONNX 验证

```bash
python python/verify_onnx.py \
  --model model/h3_next_hour_pickups.onnx \
  --schema model/feature_schema.json \
  --cases testdata/parity_test_cases.json
```

This test loads the `.onnx` file directly and does not call the original scikit-learn model.  
该测试直接加载 `.onnx` 文件，不调用原始 scikit-learn 模型。

### 3. Predict a new prepared case / 预测新的已准备样本

```bash
go run ./go \
  -model model/h3_next_hour_pickups.onnx \
  -schema model/feature_schema.json \
  -cases testdata/sample_prediction_input.json \
  -onnxruntime runtime/macos-arm64/libonnxruntime.dylib \
  -verify=false
```

The output reports the predicted completed pickups for the hour after `feature_timestamp`. Input features must follow `feature_schema.json` exactly.  
输出给出 `feature_timestamp` 后一小时的已完成 pickup 预测值。输入特征必须严格遵循 `feature_schema.json`。

## Retraining and ONNX export / 重新训练与 ONNX 导出

Retraining requires `hourly_complete_grid.pkl`, generated by the upstream hourly H3 analysis. The required columns are documented in `docs/HOURLY_CACHE_SCHEMA.md`.  
重新训练需要上游小时级 H3 分析生成的 `hourly_complete_grid.pkl`，所需字段记录在 `docs/HOURLY_CACHE_SCHEMA.md`。

```bash
python python/train_export.py \
  --input /path/to/hourly_complete_grid.pkl \
  --output model \
  --cases-output testdata/parity_test_cases.json
```

The command regenerates the ONNX model, feature schema, metrics, and fixed parity cases. Then rerun both Python and Go validation.  
该命令重新生成 ONNX 模型、特征结构、指标和固定一致性样本。完成后重新运行 Python 与 Go 验证。

## Input contract / 输入约定

Go inference receives a JSON array. Each item contains a 61-value float32 vector in the exact order defined by `model/feature_schema.json`. Missing values are filled with the training medians stored in the same schema before inference. See `testdata/INPUT_FORMAT.md`.  
Go 推理接收 JSON 数组。每条数据包含 61 个 float32 数值，顺序必须与 `model/feature_schema.json` 完全一致。推理前使用同一 schema 中保存的训练期中位数填充缺失值。详细格式见 `testdata/INPUT_FORMAT.md`。

## Reproduction on a new machine / 新机器复现条件

Inference requires only these artifacts:  
执行推理只需要以下文件：

1. `model/h3_next_hour_pickups.onnx`
2. `model/feature_schema.json`
3. A prepared JSON input with the 61 ordered features. / 一份包含 61 个有序特征的 JSON 输入。
4. Go source and a compatible ONNX Runtime library. / Go 源码和兼容的 ONNX Runtime 运行库。

Full retraining additionally requires the upstream `hourly_complete_grid.pkl` cache. The 243-million-row raw workflow is not required for inference.  
完整重新训练还需要上游 `hourly_complete_grid.pkl` 缓存；执行推理不需要重新运行 2.43 亿行的原始全量数据流程。
