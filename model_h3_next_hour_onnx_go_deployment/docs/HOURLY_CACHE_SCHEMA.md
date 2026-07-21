# Hourly Training Cache Schema / 小时级训练缓存结构

`python/train_export.py` expects a pandas pickle containing a complete H3-by-hour grid. This cache is produced by the upstream hourly H3 analysis after MatrixOne aggregation.  
`python/train_export.py` 需要一个包含完整 H3 × hour 网格的 pandas pickle。该缓存由上游小时级 H3 分析在 MatrixOne 聚合后生成。

## Required columns / 必需字段

| Column / 字段 | Meaning / 含义 |
|---|---|
| `h3` | H3 cell identifier. / H3 单元编号。 |
| `d` | Calendar date. / 日期。 |
| `timestamp` | Hour timestamp. / 小时时间戳。 |
| `hour` | Hour from 0 to 23. / 0 到 23 的小时。 |
| `dow_sun0` | Weekday with Sunday equal to 0. / 星期编码，周日为 0。 |
| `dow_mon0` | Weekday with Monday equal to 0. / 星期编码，周一为 0。 |
| `month` | Calendar month. / 月份。 |
| `day_of_year` | Day number within the year. / 年内日序号。 |
| `is_weekend` | Weekend flag. / 周末标记。 |
| `is_holiday` | US holiday flag. / 美国节假日标记。 |
| `is_dst_missing_hour` | Daylight-saving missing-hour flag. / 夏令时缺失小时标记。 |
| `pickup_count` | Completed pickups in the center H3 cell. / 中心 H3 的已完成 pickup 数。 |
| `dropoff_count` | Completed dropoffs in the center H3 cell. / 中心 H3 的已完成 dropoff 数。 |
| `neighbor_pickup_count` | Total pickups in first-ring neighboring H3 cells. / 一环周边 H3 的 pickup 总数。 |
| `active_neighbor_cells` | Number of neighboring cells with observed pickups. / 有 pickup 观测的周边单元数。 |

The grid must be sorted or sortable by `h3` and `timestamp`. Missing H3-hours should be completed before training so lag features use a continuous time axis.  
网格必须能够按 `h3` 和 `timestamp` 排序。训练前需要补全缺失的 H3-hour，确保 lag 特征使用连续时间轴。
