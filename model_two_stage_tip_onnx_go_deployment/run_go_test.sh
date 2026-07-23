#!/usr/bin/env bash
set -euo pipefail

# Resolve paths from this script so the test works from any directory.
# 根据脚本位置解析路径，因此可以从任意目录运行。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNTIME_PATH="${ONNXRUNTIME_SHARED_LIBRARY:-runtime/macos-arm64/libonnxruntime.dylib}"

go run ./go \
  -model-dir model \
  -schema model/feature_schema.json \
  -cases testdata/parity_test_cases.json \
  -onnxruntime "$RUNTIME_PATH" \
  -verify=true
