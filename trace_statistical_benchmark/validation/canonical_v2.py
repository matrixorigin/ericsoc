"""Shared helpers for canonical trace benchmark data. / 标准 Trace 数据共用工具。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TABLE_FILES = {
    "models": "models.jsonl",
    "tasks": "tasks.jsonl",
    "sessions": "sessions.jsonl",
    "traces": "traces.jsonl",
    "observations": "observations.jsonl",
    "generations": "generations.jsonl",
    "tool_calls": "tool_calls.jsonl",
    "model_answers": "model_answers.jsonl",
    "annotations": "annotations.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(canonical_json(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def parse_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def duration_ms(start_time: str, end_time: str) -> float:
    return max(0.0, (parse_time(end_time) - parse_time(start_time)).total_seconds() * 1000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundle(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {table: read_jsonl(data_dir / filename) for table, filename in TABLE_FILES.items()}
