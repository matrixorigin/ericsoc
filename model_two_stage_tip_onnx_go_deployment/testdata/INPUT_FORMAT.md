# Input format / 输入格式

The Go program reads a JSON array. Each item contains a `features` array with 17 `float32` values in the exact order stored in `model/feature_schema.json`.

Go 程序读取 JSON 数组。每条数据包含一个 `features` 数组，其中 17 个 `float32` 数值的顺序必须与 `model/feature_schema.json` 完全一致。

```json
[
  {
    "case_id": "new_trip_01",
    "trip_start_timestamp": "2024-06-01T18:00:00",
    "pickup_h3": "892664...",
    "dropoff_h3": "892664...",
    "features": [18, 5, 6, 1, 0, 10, 35, 0, 2.3, 18, 5, 0, 24, 2.8, 1.5, 7.2, 18.5]
  }
]
```

The timestamp and H3 fields are metadata for readable output. ONNX inference uses the prepared feature vector.

时间和 H3 字段用于生成容易阅读的输出；ONNX 推理实际使用已准备的特征向量。
