"""Verify ONNX outputs against stored Python references. / 使用保存的 Python 参考结果验证 ONNX 输出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    """Load the ONNX model and validate fixed parity cases. / 加载 ONNX 模型并验证固定一致性样本。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    values = np.asarray([case["features"] for case in cases], dtype=np.float32)
    expected = np.asarray(
        [case["onnx_python_prediction"] for case in cases], dtype=np.float32
    )

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    predicted = np.asarray(
        session.run([schema["output_name"]], {schema["input_name"]: values})[0]
    ).reshape(-1)
    predicted = np.clip(predicted, 0, None)
    differences = np.abs(predicted - expected)
    for case, prediction, difference in zip(cases, predicted, differences):
        print(f"{case['case_id']}: prediction={prediction:.6f}, difference={difference:.8f}")
    print(f"max difference / 最大差异: {differences.max():.10f}")
    if float(differences.max()) > args.tolerance:
        raise SystemExit("ONNX verification failed / ONNX 验证失败")
    print("PASS: ONNX Runtime predictions match the stored Python references.")
    print("通过：ONNX Runtime 预测与保存的 Python 参考结果一致。")


if __name__ == "__main__":
    main()
