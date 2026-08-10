#!/usr/bin/env python3
"""Run portable source checks for the public repository. / 对公开仓库运行可移植源码检查。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "01_data_ingestion_validation_and_h3_preparation.ipynb",
    "02_hourly_h3_demand_forecasting_and_spatial_dynamics.ipynb",
    "03_two_stage_tip_prediction_and_llm_tree_comparison.ipynb",
    "04_llm_peak_continuation_benchmark.ipynb",
    "05_chronos2_vs_lightgbm_forecasting_benchmark.ipynb",
    "06_h3_next_hour_onnx_go_deployment_validation.ipynb",
    "07_bears_event_sensitivity_and_2024_blind_detection.ipynb",
    "08_event_business_monte_carlo_simulator.ipynb",
]
MODEL_DIRS = [
    "model_h3_next_hour_onnx_go_deployment",
    "model_two_stage_tip_onnx_go_deployment",
]
REQUIRED_ROOT_FILES = [
    "README.md",
    "requirements.txt",
    "requirements-chronos.txt",
    "Chicago_TNP_Project_Report_Eric.docx",
]
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]
LOCAL_PATH_PATTERNS = ["/Users/", "/home/ec2-user/"]


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"FAIL / 失败: {message}")


def clean_code(source: str) -> str:
    """Remove Jupyter-only command lines before syntax compilation. / 编译前移除 Jupyter 专用命令行。"""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            lines.append("pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def validate_notebook(path: Path, errors: list[str]) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"{path.name}: invalid Notebook JSON ({exc})", errors)
        return

    cells = notebook.get("cells", [])
    if not cells:
        fail(f"{path.name}: no cells", errors)
        return
    first_markdown = next(
        ("".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "markdown"),
        "",
    )
    if not (re.search(r"[A-Za-z]", first_markdown) and re.search(r"[\u4e00-\u9fff]", first_markdown)):
        fail(f"{path.name}: first description is not bilingual", errors)

    for cell_number, cell in enumerate(cells, start=1):
        source = "".join(cell.get("source", []))
        if any(pattern in source for pattern in LOCAL_PATH_PATTERNS):
            fail(f"{path.name} cell {cell_number}: hard-coded local path", errors)
        if any(pattern.search(source) for pattern in SECRET_PATTERNS):
            fail(f"{path.name} cell {cell_number}: possible secret", errors)
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            fail(f"{path.name} cell {cell_number}: committed output is not empty", errors)
        if cell.get("execution_count") is not None:
            fail(f"{path.name} cell {cell_number}: execution count is not empty", errors)
        try:
            compile(clean_code(source), f"{path.name}:cell-{cell_number}", "exec")
        except SyntaxError as exc:
            fail(f"{path.name} cell {cell_number}: Python syntax error ({exc})", errors)


def validate_manifest(model_dir: Path, errors: list[str]) -> None:
    manifest = model_dir / "MANIFEST_SHA256.txt"
    if not manifest.exists():
        fail(f"{model_dir.name}: missing MANIFEST_SHA256.txt", errors)
        return
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            fail(f"{model_dir.name}: invalid manifest line {line_number}", errors)
            continue
        relative = relative.removeprefix("./")
        artifact = model_dir / relative
        if not artifact.is_file():
            fail(f"{model_dir.name}: missing manifest file {relative}", errors)
            continue
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"{model_dir.name}: checksum mismatch for {relative}", errors)


def validate_model_dir(path: Path, errors: list[str]) -> None:
    required = [
        "README.md",
        "MODEL_CARD.md",
        "go.mod",
        "go/main.go",
        "python/train_export.py",
        "python/verify_onnx.py",
        "model/feature_schema.json",
        "model/model_metrics.json",
        "testdata/parity_test_cases.json",
        "run_go_test.sh",
    ]
    for relative in required:
        if not (path / relative).is_file():
            fail(f"{path.name}: missing {relative}", errors)
    for relative in [
        "model/feature_schema.json",
        "model/model_metrics.json",
        "testdata/parity_test_cases.json",
    ]:
        artifact = path / relative
        if artifact.exists():
            try:
                json.loads(artifact.read_text(encoding="utf-8"))
            except Exception as exc:
                fail(f"{path.name}: invalid {relative} ({exc})", errors)
    validate_manifest(path, errors)


def main() -> int:
    errors: list[str] = []
    print(f"Repository / 仓库: {ROOT}")

    for relative in REQUIRED_ROOT_FILES:
        if not (ROOT / relative).is_file():
            fail(f"missing root deliverable: {relative}", errors)

    report = ROOT / "Chicago_TNP_Project_Report_Eric.docx"
    if report.exists() and not zipfile.is_zipfile(report):
        fail("client report is not a valid DOCX archive", errors)

    if (ROOT / ".Rhistory").exists():
        fail(".Rhistory must not be published", errors)

    for notebook_name in NOTEBOOKS:
        path = ROOT / notebook_name
        if not path.is_file():
            fail(f"missing notebook: {notebook_name}", errors)
        else:
            validate_notebook(path, errors)

    for directory_name in MODEL_DIRS:
        path = ROOT / directory_name
        if not path.is_dir():
            fail(f"missing model package: {directory_name}", errors)
        else:
            validate_model_dir(path, errors)

    if errors:
        print(f"\nRepository validation failed with {len(errors)} issue(s). / 仓库检查发现 {len(errors)} 个问题。")
        return 1

    print("\nPASS: repository source and artifacts are ready for review.")
    print("通过：仓库源码和模型产物已通过发布前检查。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

