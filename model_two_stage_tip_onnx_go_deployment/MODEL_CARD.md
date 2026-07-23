# Model Card: Two-Stage Recorded-Tip Prediction
# 模型卡：两阶段记录小费预测

## Intended use / 预期用途

The package provides a portable baseline for estimating the recorded tip of a completed Chicago TNP trip. It supports offline analysis, technical demonstrations, and cross-language deployment tests.

本项目提供一个可移植基线，用于估计一笔已完成 Chicago TNP 行程的记录小费。适用于离线分析、技术演示和跨语言部署测试。

## Architecture / 模型结构

The first gradient-boosting model estimates the probability of a positive recorded tip. The second model is trained only on positive-tip trips and estimates `log1p(tip)`. The final expected tip is:

第一阶段梯度提升模型估计记录正小费的概率。第二阶段模型只使用正小费行程训练，并预测 `log1p(tip)`。最终期望小费为：

```text
expected recorded tip = positive-tip probability × predicted positive-tip amount
记录小费期望值 = 正小费概率 × 预测正小费金额
```

## Features / 特征

The 17 features cover pickup/dropoff time, training-only H3 codes, route frequency, date crossing, trip duration, fare, miles, seconds, and average speed. `trip_total` is excluded because it already includes the tip.

17 个特征覆盖上下车时间、仅用训练期拟合的 H3 编码、路线频率、是否跨日、行程时长、车费、里程、秒数和平均速度。`trip_total` 已包含小费，因此被排除。

## Evaluation / 评估

The model trains on 2022-2023 and tests on 2024. The expected-tip MAE is about $1.46, compared with about $1.58 for a training-mean baseline. Full metrics are stored in `model/model_metrics.json`.

模型使用 2022-2023 训练，并在 2024 测试。期望小费 MAE 约为 1.46 美元，训练期平均值基线约为 1.58 美元。完整指标保存在 `model/model_metrics.json`。

## Limitations / 局限

- The target only includes tips recorded in the source data. / 目标只包含源数据中记录的小费。
- Unrecorded cash tips cannot be learned or evaluated. / 未记录的现金小费无法训练或评估。
- The sample uses up to 5,000 valid trips per month rather than every trip. / 样本每月最多使用 5,000 条有效行程，不是全部行程。
- Location codes and route frequencies are fitted on 2022-2023 only. / 地点编码和路线频率仅用 2022-2023 拟合。
- The model provides expected value, not a guarantee for one passenger. / 模型输出期望值，不是对单个乘客的保证。

## Deployment validation / 部署验证

Twenty fixed cases are evaluated by both Python ONNX Runtime and Go ONNX Runtime. The maximum observed cross-language difference is about $0.0000019.

20 条固定样本同时由 Python ONNX Runtime 和 Go ONNX Runtime 执行。观测到的最大跨语言差异约为 0.0000019 美元。
