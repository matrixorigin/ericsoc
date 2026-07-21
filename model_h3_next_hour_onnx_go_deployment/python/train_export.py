"""Train and export the H3 next-hour model. / 训练并导出 H3 下一小时模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from feature_engineering import FEATURES, TEST_END, TRAIN_END, build_model_frame


def patch_hist_gradient_converter_bool() -> None:
    """Normalize a converter flag across skl2onnx versions. / 统一不同 skl2onnx 版本的转换标记。"""
    import skl2onnx.common.tree_ensemble as tree_ensemble

    original = tree_ensemble.add_node

    def compatible_add_node(*args, **kwargs):
        """Convert NumPy boolean flags to Python integers. / 将 NumPy 布尔标记转换为 Python 整数。"""
        if "nodes_missing_value_tracks_true" in kwargs:
            kwargs["nodes_missing_value_tracks_true"] = int(
                bool(kwargs["nodes_missing_value_tracks_true"])
            )
        return original(*args, **kwargs)

    tree_ensemble.add_node = compatible_add_node


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics after nonnegative clipping. / 对非负截断后的预测计算回归指标。"""
    prediction = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    truth = np.asarray(y_true, dtype=float)
    mae = mean_absolute_error(truth, prediction)
    return {
        "mae": float(mae),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "mean_actual": float(truth.mean()),
        "mae_pct_of_mean": float(mae / truth.mean() * 100),
    }


def sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of one file. / 计算一个文件的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Train, export, validate, and write deployment artifacts. / 训练、导出、验证并写入部署产物。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=180)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.cases_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hourly = pd.read_pickle(args.input)
    frame, cell_codes = build_model_frame(hourly)

    valid = (
        frame["target_pickup_h1"].notna()
        & (frame["is_dst_missing_hour"] == 0)
        & (frame["target_dst_h1"].fillna(0) == 0)
    )
    train_mask = valid & (frame["target_timestamp_h1"] < TRAIN_END)
    test_mask = (
        valid
        & (frame["target_timestamp_h1"] >= TRAIN_END)
        & (frame["target_timestamp_h1"] < TEST_END)
    )
    train_index = frame.index[train_mask]
    test_index = frame.index[test_mask]

    clean = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = clean.loc[train_index].median().fillna(0.0)
    x_train = clean.loc[train_index].fillna(medians).to_numpy(dtype=np.float32)
    x_test = clean.loc[test_index].fillna(medians).to_numpy(dtype=np.float32)
    y_train = frame.loc[train_index, "target_pickup_h1"].to_numpy(dtype=np.float32)
    y_test = frame.loc[test_index, "target_pickup_h1"].to_numpy(dtype=np.float32)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=args.max_iter,
        learning_rate=0.06,
        max_leaf_nodes=63,
        min_samples_leaf=80,
        l2_regularization=1.0,
        random_state=42,
    )
    fit_started = time.time()
    model.fit(x_train, y_train)
    fit_seconds = time.time() - fit_started
    sklearn_prediction = np.clip(model.predict(x_test), 0, None).astype(np.float32)

    model_path = args.output / "h3_next_hour_pickups.onnx"
    patch_hist_gradient_converter_bool()
    onnx_model = convert_sklearn(
        model,
        name="Chicago TNP H3 next-hour pickup forecast",
        initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
        target_opset={"": 18, "ai.onnx.ml": 3},
    )
    onnx.checker.check_model(onnx_model)
    model_path.write_bytes(onnx_model.SerializeToString())

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_prediction = np.asarray(
        session.run([output_name], {input_name: x_test})[0]
    ).reshape(-1)
    onnx_prediction = np.clip(onnx_prediction, 0, None).astype(np.float32)
    parity_difference = np.abs(
        sklearn_prediction.astype(np.float64) - onnx_prediction.astype(np.float64)
    )

    naive_current = frame.loc[test_index, "naive_current"].to_numpy(dtype=float)
    naive_weekly = frame.loc[test_index, "naive_weekly_h1"].to_numpy(dtype=float)
    report = {
        "task": "predict pickup count for the next hour in the same H3 cell",
        "train_target_period": "2022-01-01 through 2023-12-31",
        "test_target_period": "2024-01-01 through 2024-12-31",
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "feature_count": len(FEATURES),
        "fit_seconds": fit_seconds,
        "model": metrics(y_test, onnx_prediction),
        "naive_current_hour": metrics(y_test, naive_current),
        "naive_last_week": metrics(y_test, naive_weekly),
        "python_onnx_parity": {
            "max_absolute_difference": float(parity_difference.max()),
            "mean_absolute_difference": float(parity_difference.mean()),
            "within_1e-4_count": int((parity_difference <= 1e-4).sum()),
            "total_count": int(len(parity_difference)),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "total_seconds": time.time() - started,
    }

    schema = {
        "schema_version": 1,
        "model_file": model_path.name,
        "input_name": input_name,
        "output_name": output_name,
        "input_dtype": "float32",
        "input_shape": ["batch", len(FEATURES)],
        "feature_order": FEATURES,
        "feature_medians": {name: float(medians[name]) for name in FEATURES},
        "cell_codes": cell_codes,
        "postprocessing": {"clip_prediction_min": 0.0},
        "target": "pickup_count in the same H3 cell one hour after feature_timestamp",
    }

    positions = np.linspace(0, len(test_index) - 1, num=20, dtype=int)
    cases = []
    for number, position in enumerate(positions, start=1):
        row_index = int(test_index[position])
        vector = clean.loc[row_index].fillna(medians).to_numpy(dtype=np.float32)
        cases.append(
            {
                "case_id": f"case_{number:02d}",
                "h3": str(frame.at[row_index, "h3"]),
                "feature_timestamp": frame.at[row_index, "timestamp"].isoformat(),
                "target_timestamp": frame.at[row_index, "target_timestamp_h1"].isoformat(),
                "features": [float(value) for value in vector],
                "actual_pickups": float(frame.at[row_index, "target_pickup_h1"]),
                "python_prediction": float(sklearn_prediction[position]),
                "onnx_python_prediction": float(onnx_prediction[position]),
            }
        )

    schema_path = args.output / "feature_schema.json"
    metrics_path = args.output / "model_metrics.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    args.cases_output.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    report["artifacts"] = {
        "model_sha256": sha256(model_path),
        "schema_sha256": sha256(schema_path),
        "test_cases_sha256": sha256(args.cases_output),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
