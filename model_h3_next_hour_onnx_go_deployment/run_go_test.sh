#!/usr/bin/env bash
set -euo pipefail

# Resolve paths from the project directory. / 根据项目目录解析路径。
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v go >/dev/null 2>&1; then
  echo "Go 1.24+ is required. / 需要安装 Go 1.24 或更高版本。" >&2
  exit 1
fi

RUNTIME_PATH="${ONNXRUNTIME_SHARED_LIBRARY:-$ROOT_DIR/runtime/macos-arm64/libonnxruntime.dylib}"
if [[ ! -f "$RUNTIME_PATH" ]]; then
  echo "ONNX Runtime library not found: $RUNTIME_PATH" >&2
  echo "未找到 ONNX Runtime 动态库。请设置 ONNXRUNTIME_SHARED_LIBRARY。" >&2
  exit 1
fi

go run ./go \
  -model "$ROOT_DIR/model/h3_next_hour_pickups.onnx" \
  -schema "$ROOT_DIR/model/feature_schema.json" \
  -cases "$ROOT_DIR/testdata/parity_test_cases.json" \
  -onnxruntime "$RUNTIME_PATH" \
  -tolerance 0.001 \
  -verify=true
