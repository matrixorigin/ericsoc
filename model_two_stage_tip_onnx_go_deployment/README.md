# Chicago TNP Two-Stage Recorded-Tip Model: ONNX and Go Deployment
# Chicago TNP 两阶段记录小费模型：ONNX 与 Go 部署

## Project overview / 项目概览

This package trains a two-stage model in Python, exports both stages to ONNX, and runs the same calculation in Go. Given 17 prepared features for one completed trip, the model returns:

本项目在 Python 中训练两阶段模型，将两个阶段导出为 ONNX，并在 Go 中执行同样的计算。输入一笔已完成行程的 17 个已准备特征，模型输出：

1. Probability that a positive tip is recorded. / 记录正小费的概率。
2. Predicted amount when a positive tip is recorded. / 在记录正小费时的预测金额。
3. Expected recorded tip: probability multiplied by positive-tip amount. / 记录小费期望值：概率乘以正小费金额。

```text
Prepared trip features
        ↓
ONNX presence classifier ──→ probability of a positive recorded tip
        ↓
ONNX amount regressor ─────→ positive-tip amount
        ↓
probability × amount ──────→ expected recorded tip
```

```text
已准备的行程特征
        ↓
ONNX 小费分类器 ──→ 记录正小费的概率
        ↓
ONNX 金额回归器 ──→ 正小费金额
        ↓
概率 × 金额 ─────→ 记录小费期望值
```

The target is the `tip` field recorded in completed-trip data. Cash tips that were not recorded are outside this dataset.

预测目标是已完成行程数据中记录的 `tip` 字段。数据中未记录的现金小费不在本模型范围内。

## Repository contents / 目录内容

| Path / 路径 | Purpose / 用途 |
|---|---|
| `two_stage_tip_onnx_training_and_go_validation.ipynb` | Reproducible training, export, and validation workflow. / 可复现的训练、导出与验证流程。 |
| `python/feature_engineering.py` | Rebuilds the 17 leakage-safe features. / 重新构造 17 个避免信息泄漏的特征。 |
| `python/train_export.py` | Trains both stages and exports ONNX artifacts. / 训练两个阶段并导出 ONNX 产物。 |
| `python/verify_onnx.py` | Verifies Python ONNX predictions. / 验证 Python ONNX 预测。 |
| `go/main.go` | Loads both ONNX models and calculates expected tip. / 加载两个 ONNX 模型并计算期望小费。 |
| `model/tip_presence_classifier.onnx` | Stage 1: positive recorded-tip probability. / 第一阶段：记录正小费概率。 |
| `model/positive_tip_amount_regressor.onnx` | Stage 2: positive-tip amount. / 第二阶段：正小费金额。 |
| `model/feature_schema.json` | Feature order, medians, category mappings, and model contract. / 特征顺序、中位数、类别映射和模型约定。 |
| `model/model_metrics.json` | 2024 test metrics and Python-to-ONNX parity. / 2024 测试指标与 Python-ONNX 一致性结果。 |
| `testdata/parity_test_cases.json` | Twenty fixed cross-language validation cases. / 20 条固定跨语言验证样本。 |
| `testdata/sample_prediction_input.json` | Example inputs without reference answers. / 不含参考答案的示例输入。 |
| `runtime/macos-arm64/libonnxruntime.dylib` | Included Apple Silicon demonstration runtime. / 随包提供的 Apple Silicon 演示运行库。 |

## Model and result summary / 模型与结果概览

- Models: `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`. / 模型：`HistGradientBoostingClassifier` 与 `HistGradientBoostingRegressor`。
- Training period: 2022-2023; test period: 2024. / 训练期：2022-2023；测试期：2024。
- Training rows: 119,468 trips; test rows: 59,656 trips. / 训练样本：119,468 条行程；测试样本：59,656 条行程。
- Feature count: 17. / 特征数量：17。
- Recorded-tip classification ROC-AUC: 0.6425. / 记录小费分类 ROC-AUC：0.6425。
- Expected-tip MAE: $1.46, compared with $1.58 for the training-mean baseline. / 期望小费 MAE：1.46 美元；训练期平均值基线为 1.58 美元。
- Expected-tip R²: 0.1344. / 期望小费 R²：0.1344。
- Go/Python validation: maximum difference across 20 fixed cases is about $0.0000019. / Go/Python 验证：20 条固定样本最大差异约为 0.0000019 美元。

The model beats the simple mean baseline, but recorded tipping remains difficult to predict from trip fields alone. This package is a portable baseline, not a claim that each individual tip can be predicted exactly.

模型优于简单平均值基线，但仅凭行程字段仍然很难准确预测记录小费。本项目提供的是可移植基线，不代表能够精确预测每一笔小费。

## Environment / 环境配置

### Go inference / Go 推理

- Go 1.24 or newer. / Go 1.24 或更高版本。
- A compatible ONNX Runtime shared library. / 兼容的 ONNX Runtime 动态库。
- The included runtime supports macOS Apple Silicon. / 随包运行库支持 macOS Apple Silicon。

For another operating system, install the matching ONNX Runtime library and set:

其他操作系统需要安装对应的 ONNX Runtime，并设置：

```bash
export ONNXRUNTIME_SHARED_LIBRARY="/absolute/path/to/libonnxruntime.so"
```

Use `.dylib` on macOS, `.so` on Linux, and `.dll` on Windows.

macOS 使用 `.dylib`，Linux 使用 `.so`，Windows 使用 `.dll`。

### Python validation or retraining / Python 验证或重新训练

Python 3.12 is the validated environment.

已验证的环境为 Python 3.12。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MatrixOne is not required for inference. It is only used upstream to create the sampled training cache.

模型推理不需要 MatrixOne。MatrixOne 仅在上游生成抽样训练缓存时使用。

## Quick demonstration / 快速演示

Run commands from this directory.

在本目录中运行以下命令。

### 1. Go and Python parity test / Go 与 Python 一致性测试

```bash
./run_go_test.sh
```

The script compiles the Go code, loads both ONNX files, predicts the same 20 cases used by Python ONNX Runtime, and checks a tolerance of $0.001. A successful run ends with:

该脚本会编译 Go 代码、加载两个 ONNX 文件、预测 Python ONNX Runtime 使用的相同 20 条样本，并检查 0.001 美元的误差容限。成功时最后显示：

```text
PASS: Go and Python ONNX Runtime predictions match within tolerance.
```

This confirms that the Python-exported models produce operationally identical outputs in Go.

这说明 Python 导出的模型在 Go 中能够产生业务上相同的输出。

### 2. Python ONNX validation / Python ONNX 验证

```bash
python python/verify_onnx.py \
  --model-dir model \
  --schema model/feature_schema.json \
  --cases testdata/parity_test_cases.json
```

This loads the ONNX files directly and does not call the original scikit-learn models.

该命令直接加载 ONNX 文件，不调用原始 scikit-learn 模型。

### 3. Predict prepared trips / 预测已准备的行程

```bash
go run ./go \
  -model-dir model \
  -schema model/feature_schema.json \
  -cases testdata/sample_prediction_input.json \
  -onnxruntime runtime/macos-arm64/libonnxruntime.dylib \
  -verify=false
```

The output reports the tip probability, positive-tip amount, and expected recorded tip for each prepared trip.

输出会给出每条已准备行程的记录小费概率、正小费金额和记录小费期望值。

## Retraining and ONNX export / 重新训练与 ONNX 导出

Retraining uses the cache generated by `02_two_stage_tip_amount_prediction.ipynb`.

重新训练使用 `02_two_stage_tip_amount_prediction.ipynb` 生成的缓存。

```bash
python python/train_export.py \
  --input /path/to/tip_model_sample_5000_per_month.csv.gz \
  --output model \
  --cases-output testdata/parity_test_cases.json
```

After retraining, rerun both Python and Go parity tests.

重新训练后，需要再次运行 Python 和 Go 一致性测试。

## Input contract / 输入约定

Go inference receives a JSON array. Each item contains a 17-value `float32` vector in the exact order defined by `model/feature_schema.json`. See `testdata/INPUT_FORMAT.md`.

Go 推理接收 JSON 数组。每条数据包含 17 个 `float32` 数值，顺序必须与 `model/feature_schema.json` 完全一致。详细格式见 `testdata/INPUT_FORMAT.md`。

## Reproduction on a new machine / 新机器复现

Inference requires only:

执行推理只需要：

1. The two ONNX model files. / 两个 ONNX 模型文件。
2. `model/feature_schema.json`.
3. A prepared JSON input with 17 ordered features. / 一份包含 17 个有序特征的 JSON 输入。
4. The Go source and a compatible ONNX Runtime library. / Go 源码和兼容的 ONNX Runtime 运行库。

Full retraining additionally requires the sampled tip cache. The 243-million-row source table is not required for inference.

完整重新训练还需要小费抽样缓存；执行推理不需要重新处理 2.43 亿行源数据。
