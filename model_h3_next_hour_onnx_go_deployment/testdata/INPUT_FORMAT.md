# Inference Input Format / 推理输入格式

The Go program reads a JSON array. Each object must contain a case identifier, an H3 cell, a feature timestamp, and exactly 61 float32-compatible values.  
Go 程序读取一个 JSON 数组。每个对象必须包含案例编号、H3 单元、特征时间，以及恰好 61 个可转换为 float32 的数值。

```json
{
  "case_id": "new_case_01",
  "h3": "89275934ed3ffff",
  "feature_timestamp": "2024-01-01T05:00:00",
  "features": [61 ordered numeric values]
}
```

```text
features 必须包含 61 个按顺序排列的数值。
```

The `features` array must follow `model/feature_schema.json -> feature_order` exactly. Before inference, missing values must be filled with the corresponding values in `feature_medians`. The H3 cell must exist in `cell_codes`.  
`features` 数组必须严格遵循 `model/feature_schema.json -> feature_order`。推理前，缺失值必须使用 `feature_medians` 中对应数值填充，H3 单元必须存在于 `cell_codes`。

`parity_test_cases.json` also includes `onnx_python_prediction` for cross-language verification. Production-style input can omit reference fields and use `-verify=false`.  
`parity_test_cases.json` 还包含用于跨语言验证的 `onnx_python_prediction`。普通预测输入可以省略参考字段，并使用 `-verify=false`。
