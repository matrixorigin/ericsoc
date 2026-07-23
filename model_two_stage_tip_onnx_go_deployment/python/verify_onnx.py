"""Verify the two-stage ONNX model in Python. / 在 Python 中验证两阶段 ONNX 模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    """Run fixed cases and compare stored ONNX references. / 运行固定样本并对比已保存 ONNX 参考值。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.001)
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    features = np.asarray([case["features"] for case in cases], dtype=np.float32)

    presence = schema["presence_model"]
    amount = schema["amount_model"]
    presence_session = ort.InferenceSession(
        str(args.model_dir / presence["file"]), providers=["CPUExecutionProvider"]
    )
    amount_session = ort.InferenceSession(
        str(args.model_dir / amount["file"]), providers=["CPUExecutionProvider"]
    )

    matrix = np.asarray(
        presence_session.run(
            [presence["probability_output_name"]],
            {presence["input_name"]: features},
        )[0]
    )
    probability = matrix[:, int(presence["positive_class_index"])]
    log_amount = np.asarray(
        amount_session.run(
            [amount["output_name"]], {amount["input_name"]: features}
        )[0]
    ).reshape(-1)
    tip_cap = float(schema["postprocessing"]["tip_cap"])
    positive_amount = np.clip(np.expm1(log_amount), 0, tip_cap)
    expected_tip = probability * positive_amount

    reference_probability = np.asarray(
        [case["onnx_python_tip_probability"] for case in cases]
    )
    reference_amount = np.asarray(
        [case["onnx_python_positive_tip_amount"] for case in cases]
    )
    reference_expected = np.asarray(
        [case["onnx_python_expected_tip"] for case in cases]
    )
    differences = {
        "tip_probability": float(
            np.max(np.abs(probability - reference_probability))
        ),
        "positive_tip_amount": float(
            np.max(np.abs(positive_amount - reference_amount))
        ),
        "expected_tip": float(np.max(np.abs(expected_tip - reference_expected))),
    }
    print(json.dumps(differences, indent=2))
    if max(differences.values()) > args.tolerance:
        raise SystemExit("FAIL: Python ONNX parity exceeds tolerance. / 一致性误差超过容限。")
    print("PASS: Python ONNX predictions match stored references.")
    print("通过：Python ONNX 预测与已保存参考结果一致。")


if __name__ == "__main__":
    main()
