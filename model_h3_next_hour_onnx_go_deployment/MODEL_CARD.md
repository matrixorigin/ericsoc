# Model Card: H3 Next-Hour Completed Pickup Forecast  
# 模型说明：H3 下一小时已完成上车订单预测

## Intended task / 任务定义

At the end of an hour, predict completed pickup trips in the same H3 cell during the next hour. The model consumes hourly aggregate features rather than individual trip records.  
在一个小时结束时，预测同一 H3 单元下一小时的已完成上车订单数。模型读取小时级聚合特征，而不是单笔行程记录。

## Data and time split / 数据与时间切分

- Spatial scope: top 50 pickup H3 cells selected by the upstream analysis. / 空间范围：上游分析选择的 Top 50 pickup H3 单元。
- Training targets: 2022-01-01 through 2023-12-31. / 训练目标期：2022-01-01 至 2023-12-31。
- Test targets: 2024-01-01 through 2024-12-31. / 测试目标期：2024-01-01 至 2024-12-31。
- Training rows: 875,750 H3-hours. / 训练样本：875,750 个 H3-hour。
- Test rows: 439,100 H3-hours. / 测试样本：439,100 个 H3-hour。

The split follows time order. Features at time `t` use only information available at or before `t` to predict `t+1`, preventing future-data leakage.  
数据按时间切分。时间 `t` 的特征只使用 `t` 或更早的信息预测 `t+1`，避免未来数据泄漏。

## Model structure / 模型结构

The estimator is `HistGradientBoostingRegressor` with absolute-error loss, 180 boosting iterations, learning rate 0.06, up to 63 leaf nodes, minimum 80 samples per leaf, and L2 regularization 1.0.  
模型为使用绝对误差损失的 `HistGradientBoostingRegressor`，包含 180 次提升迭代、0.06 学习率、最多 63 个叶节点、每个叶节点至少 80 条样本，以及 1.0 的 L2 正则化。

The tree model captures nonlinear interactions among recent demand, calendar position, pickup/dropoff flow, and neighboring H3 activity. Absolute-error loss limits the influence of a small number of extreme peaks. The trained estimator is exported to ONNX for language-independent inference.  
树模型能够学习近期需求、日历位置、上下车流动和周边 H3 活跃程度之间的非线性关系。绝对误差损失限制少量极端高峰对训练的影响。训练完成后模型导出为 ONNX，实现跨语言推理。

## Feature groups / 特征组

The 61 features follow one fixed order and belong to four groups:  
61 个特征使用固定顺序，分成四组：

1. Calendar and location: H3 code, hour, weekday, month, holiday, and cyclical encodings. / 日历与位置：H3 编码、小时、星期、月份、节假日和周期编码。
2. Local demand history: current pickup, 1/2/3/6/24/168-hour lags, rolling statistics, expected demand, z-score, velocity, and acceleration. / 本地需求历史：当前 pickup、多种 lag、滚动统计、期望需求、z-score、速度和加速度。
3. Pickup/dropoff flow: dropoff history, net flow, and dropoff-to-pickup ratio. / 上下车流动：dropoff 历史、净流量和 dropoff/pickup 比率。
4. First-ring H3 context: neighboring pickup totals, active neighbors, expected values, z-scores, and lags. / H3 一环环境：周边 pickup 总量、活跃邻居、期望值、z-score 和 lag。

`model/feature_schema.json` is the deployment contract for order, medians, H3 codes, tensor names, and postprocessing.  
`model/feature_schema.json` 是部署约定，定义特征顺序、中位数、H3 编码、张量名称和后处理规则。

## 2024 test results / 2024 测试结果

| Method / 方法 | MAE | RMSE | R² |
|---|---:|---:|---:|
| ONNX gradient-boosting model / ONNX 梯度提升模型 | 12.66 | 24.39 | 0.9412 |
| Current-hour baseline / 当前小时基线 | 20.05 | 36.20 | 0.8705 |
| Last-week same-hour baseline / 上周同期基线 | 19.20 | 40.60 | 0.8371 |

MAE 12.66 means that across the top 50 H3 cells in 2024, the prediction differs from actual completed pickups by about 12.66 trips per H3-hour on average. Relative to the current-hour baseline, MAE decreases by about 36.9%; relative to the last-week baseline, it decreases by about 34.1%.  
MAE 12.66 表示在 2024 年 Top 50 H3 单元中，每个 H3-hour 的预测与实际已完成 pickup 平均相差约 12.66 单。与当前小时基线相比，MAE 下降约 36.9%；与上周同期基线相比，下降约 34.1%。

## Cross-language parity / 跨语言一致性

Across all 439,100 test rows, scikit-learn and Python ONNX Runtime differ by at most 0.000244 pickups. Across the 20 fixed deployment cases, Go and Python ONNX Runtime differ by at most 0.000183 pickups. The acceptance tolerance is 0.001.  
在全部 439,100 条测试样本上，scikit-learn 与 Python ONNX Runtime 最大相差 0.000244 个 pickup。在 20 条固定部署样本上，Go 与 Python ONNX Runtime 最大相差 0.000183 个 pickup。验收容限为 0.001。

## Limitations / 使用限制

- The target is completed pickups, not all ride requests or unmet demand. / 目标是已完成 pickup，不是全部叫车请求或未满足需求。
- The model covers the top 50 H3 cells represented in the training schema. / 模型覆盖训练 schema 中的 Top 50 H3 单元。
- Go inference expects prepared features; online MatrixOne aggregation is outside this deployment example. / Go 推理接收已准备特征，在线 MatrixOne 聚合不在本部署示例范围内。
- External events, weather, and road closures are not included in this model version. / 本模型未加入活动、天气和道路封闭等外部变量。
