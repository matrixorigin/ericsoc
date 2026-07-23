"""Train and export the two-stage recorded-tip model. / 训练并导出两阶段记录小费模型。"""

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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from feature_engineering import FEATURES, prepare_training_data, preprocessing_json


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


def sha256(path: Path) -> str:
    """Calculate one file's SHA-256 digest. / 计算单个文件的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate nonnegative regression metrics. / 计算非负回归指标。"""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "mean_actual": float(truth.mean()),
    }


def find_probability_output(session: ort.InferenceSession) -> tuple[str, int]:
    """Find the dense binary probability output and positive-class index. / 找到二分类概率输出及正类位置。"""
    for output in session.get_outputs():
        shape = output.shape
        if len(shape) == 2 and shape[-1] == 2:
            return output.name, 1
    raise RuntimeError("Dense two-class probability output was not found.")


def run_onnx(
    presence_path: Path,
    amount_path: Path,
    features: np.ndarray,
    tip_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str | int]]:
    """Run both ONNX stages and combine their outputs. / 运行两个 ONNX 阶段并组合输出。"""
    presence_session = ort.InferenceSession(
        str(presence_path), providers=["CPUExecutionProvider"]
    )
    amount_session = ort.InferenceSession(
        str(amount_path), providers=["CPUExecutionProvider"]
    )
    presence_input = presence_session.get_inputs()[0].name
    amount_input = amount_session.get_inputs()[0].name
    probability_output, positive_index = find_probability_output(presence_session)
    amount_output = amount_session.get_outputs()[0].name

    probability_matrix = np.asarray(
        presence_session.run([probability_output], {presence_input: features})[0]
    )
    probability = np.clip(probability_matrix[:, positive_index], 0, 1)
    log_amount = np.asarray(
        amount_session.run([amount_output], {amount_input: features})[0]
    ).reshape(-1)
    positive_amount = np.clip(np.expm1(log_amount), 0, tip_cap)
    expected_tip = probability * positive_amount
    contract = {
        "presence_input_name": presence_input,
        "presence_probability_output_name": probability_output,
        "positive_class_index": positive_index,
        "amount_input_name": amount_input,
        "amount_output_name": amount_output,
    }
    return probability, positive_amount, expected_tip, contract


def main() -> None:
    """Train, export, validate, and write deployment artifacts. / 训练、导出、验证并写入部署产物。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=220)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.cases_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    raw = pd.read_csv(
        args.input,
        compression="infer",
        parse_dates=["trip_start_timestamp", "trip_end_timestamp"],
    )
    prepared = prepare_training_data(raw)
    train = prepared.train
    test = prepared.test

    clean_train = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    clean_test = test[FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = clean_train.median().fillna(0.0)
    x_train = clean_train.fillna(medians).to_numpy(dtype=np.float32)
    x_test = clean_test.fillna(medians).to_numpy(dtype=np.float32)
    y_presence = train["has_tip"].to_numpy(dtype=np.int64)
    y_test_presence = test["has_tip"].to_numpy(dtype=np.int64)
    positive_train = train["tip"].gt(0).to_numpy()
    y_log_amount = np.log1p(
        train.loc[positive_train, "tip"].to_numpy(dtype=np.float32)
    )

    presence_model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=args.max_iter,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=42,
    )
    amount_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=args.max_iter,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=42,
    )

    fit_started = time.time()
    presence_model.fit(x_train, y_presence)
    amount_model.fit(x_train[positive_train], y_log_amount)
    fit_seconds = time.time() - fit_started

    sklearn_probability = presence_model.predict_proba(x_test)[:, 1]
    sklearn_positive_amount = np.clip(
        np.expm1(amount_model.predict(x_test)), 0, prepared.tip_cap
    )
    sklearn_expected_tip = sklearn_probability * sklearn_positive_amount

    patch_hist_gradient_converter_bool()
    initial_types = [("features", FloatTensorType([None, len(FEATURES)]))]
    presence_onnx = convert_sklearn(
        presence_model,
        name="Chicago TNP recorded-tip presence",
        initial_types=initial_types,
        options={id(presence_model): {"zipmap": False}},
        target_opset={"": 18, "ai.onnx.ml": 3},
    )
    amount_onnx = convert_sklearn(
        amount_model,
        name="Chicago TNP positive recorded-tip amount",
        initial_types=initial_types,
        target_opset={"": 18, "ai.onnx.ml": 3},
    )
    onnx.checker.check_model(presence_onnx)
    onnx.checker.check_model(amount_onnx)

    presence_path = args.output / "tip_presence_classifier.onnx"
    amount_path = args.output / "positive_tip_amount_regressor.onnx"
    presence_path.write_bytes(presence_onnx.SerializeToString())
    amount_path.write_bytes(amount_onnx.SerializeToString())

    (
        onnx_probability,
        onnx_positive_amount,
        onnx_expected_tip,
        contract,
    ) = run_onnx(presence_path, amount_path, x_test, prepared.tip_cap)

    predicted_presence = onnx_probability >= 0.5
    positive_test = test["tip"].gt(0).to_numpy()
    baseline_expected = np.full(
        len(test), train["tip"].mean(), dtype=np.float32
    )
    report = {
        "task": "two-stage expected recorded-tip prediction",
        "target_note": "tip is the recorded field in completed-trip data; unrecorded cash tips are not observed",
        "train_period": "2022-01-01 through 2023-12-31",
        "test_period": "2024-01-01 through 2024-12-31",
        "train_rows": int(len(train)),
        "positive_tip_train_rows": int(positive_train.sum()),
        "test_rows": int(len(test)),
        "feature_count": len(FEATURES),
        "tip_cap": float(prepared.tip_cap),
        "fit_seconds": fit_seconds,
        "classification": {
            "roc_auc": float(roc_auc_score(y_test_presence, onnx_probability)),
            "pr_auc": float(
                average_precision_score(y_test_presence, onnx_probability)
            ),
            "brier": float(brier_score_loss(y_test_presence, onnx_probability)),
            "precision_at_0_5": float(
                precision_score(
                    y_test_presence, predicted_presence, zero_division=0
                )
            ),
            "recall_at_0_5": float(
                recall_score(y_test_presence, predicted_presence, zero_division=0)
            ),
            "f1_at_0_5": float(
                f1_score(y_test_presence, predicted_presence, zero_division=0)
            ),
        },
        "expected_tip_model": regression_metrics(
            test["tip"].to_numpy(), onnx_expected_tip
        ),
        "training_mean_baseline": regression_metrics(
            test["tip"].to_numpy(), baseline_expected
        ),
        "positive_tip_amount_mae": float(
            mean_absolute_error(
                test.loc[positive_test, "tip"],
                onnx_positive_amount[positive_test],
            )
        ),
        "python_onnx_parity": {
            "probability_max_absolute_difference": float(
                np.max(np.abs(sklearn_probability - onnx_probability))
            ),
            "positive_amount_max_absolute_difference": float(
                np.max(np.abs(sklearn_positive_amount - onnx_positive_amount))
            ),
            "expected_tip_max_absolute_difference": float(
                np.max(np.abs(sklearn_expected_tip - onnx_expected_tip))
            ),
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
        "feature_order": FEATURES,
        "feature_count": len(FEATURES),
        "input_dtype": "float32",
        "input_shape": ["batch", len(FEATURES)],
        "feature_medians": {name: float(medians[name]) for name in FEATURES},
        "presence_model": {
            "file": presence_path.name,
            "input_name": contract["presence_input_name"],
            "probability_output_name": contract[
                "presence_probability_output_name"
            ],
            "positive_class_index": contract["positive_class_index"],
        },
        "amount_model": {
            "file": amount_path.name,
            "input_name": contract["amount_input_name"],
            "output_name": contract["amount_output_name"],
            "prediction_scale": "log1p_tip",
        },
        "postprocessing": {
            "positive_tip_amount": "clip(expm1(amount_model_output), 0, tip_cap)",
            "expected_tip": "positive_tip_probability * positive_tip_amount",
            "tip_cap": float(prepared.tip_cap),
        },
        "target": "expected recorded tip for one completed trip",
        "preprocessing": preprocessing_json(prepared),
    }

    positions = np.linspace(0, len(test) - 1, num=20, dtype=int)
    cases = []
    for number, position in enumerate(positions, start=1):
        row = test.iloc[int(position)]
        vector = clean_test.iloc[int(position)].fillna(medians).to_numpy(
            dtype=np.float32
        )
        cases.append(
            {
                "case_id": f"tip_case_{number:02d}",
                "trip_start_timestamp": row["trip_start_timestamp"].isoformat(),
                "pickup_h3": str(row["pickup_h3"]),
                "dropoff_h3": str(row["dropoff_h3"]),
                "features": [float(value) for value in vector],
                "actual_recorded_tip": float(row["tip"]),
                "onnx_python_tip_probability": float(
                    onnx_probability[int(position)]
                ),
                "onnx_python_positive_tip_amount": float(
                    onnx_positive_amount[int(position)]
                ),
                "onnx_python_expected_tip": float(
                    onnx_expected_tip[int(position)]
                ),
            }
        )

    schema_path = args.output / "feature_schema.json"
    metrics_path = args.output / "model_metrics.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    args.cases_output.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    report["artifacts"] = {
        "presence_model_sha256": sha256(presence_path),
        "amount_model_sha256": sha256(amount_path),
        "schema_sha256": sha256(schema_path),
        "test_cases_sha256": sha256(args.cases_output),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
