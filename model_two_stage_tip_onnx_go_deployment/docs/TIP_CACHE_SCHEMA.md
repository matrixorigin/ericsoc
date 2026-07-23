# Tip training cache / 小费训练缓存

Retraining uses the sampled CSV cache produced by `02_two_stage_tip_amount_prediction.ipynb`.

重新训练使用 `02_two_stage_tip_amount_prediction.ipynb` 生成的 CSV 抽样缓存。

Required columns / 必需字段：

- `trip_start_timestamp`
- `trip_end_timestamp`
- `pickup_h3`
- `dropoff_h3`
- `trip_seconds`
- `trip_miles`
- `fare`
- `tip`

The deployment model does not use `trip_total`, because it already contains the tip and would leak the target.

部署模型不使用 `trip_total`，因为该字段已经包含小费，会造成目标泄漏。
