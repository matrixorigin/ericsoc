#!/usr/bin/env python3
"""Validate synthetic trace tables and report whether they support the benchmark."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TABLES = ("models", "tasks", "sessions", "traces", "observations", "generations", "tool_calls", "model_answers", "annotations")
PRIMARY_KEYS = {
    "models": "model_id",
    "tasks": "task_id",
    "sessions": "session_id",
    "traces": "trace_id",
    "observations": "observation_id",
    "generations": "observation_id",
    "tool_calls": "observation_id",
    "model_answers": "answer_id",
    "annotations": "annotation_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{number}")
            rows.append(row)
    return rows


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate(data_dir: Path) -> dict[str, Any]:
    tables = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in TABLES}
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    for name, key in PRIMARY_KEYS.items():
        values = [row.get(key) for row in tables[name]]
        checks[f"{name}_primary_keys_present"] = all(isinstance(value, str) and value for value in values)
        checks[f"{name}_primary_keys_unique"] = len(values) == len(set(values))

    model_ids = {row["model_id"] for row in tables["models"]}
    task_ids = {row["task_id"] for row in tables["tasks"]}
    session_ids = {row["session_id"] for row in tables["sessions"]}
    trace_ids = {row["trace_id"] for row in tables["traces"]}
    observation_ids = {row["observation_id"] for row in tables["observations"]}
    observations = {row["observation_id"]: row for row in tables["observations"]}

    checks["trace_foreign_keys_resolve"] = all(
        row["model_id"] in model_ids and row["task_id"] in task_ids and row["session_id"] in session_ids
        for row in tables["traces"]
    )
    checks["observation_trace_links_resolve"] = all(row["trace_id"] in trace_ids for row in tables["observations"])
    checks["observation_parent_links_resolve"] = all(
        row.get("parent_observation_id") is None or row["parent_observation_id"] in observation_ids
        for row in tables["observations"]
    )
    roots = Counter(row["trace_id"] for row in tables["observations"] if row.get("parent_observation_id") is None)
    checks["exactly_one_root_per_trace"] = all(roots[trace_id] == 1 for trace_id in trace_ids)
    checks["generation_rows_match_observations"] = all(
        row["observation_id"] in observations
        and observations[row["observation_id"]]["trace_id"] == row["trace_id"]
        and observations[row["observation_id"]]["type"] == "generation"
        for row in tables["generations"]
    )
    checks["tool_rows_match_observations"] = all(
        row["observation_id"] in observations
        and observations[row["observation_id"]]["trace_id"] == row["trace_id"]
        and observations[row["observation_id"]]["type"] == "tool"
        for row in tables["tool_calls"]
    )
    checks["answer_foreign_keys_resolve"] = all(
        row["trace_id"] in trace_ids and row["task_id"] in task_ids and row["model_id"] in model_ids
        for row in tables["model_answers"]
    )

    trace_times_ok = all(parse_time(row["start_time"]) <= parse_time(row["end_time"]) for row in tables["traces"])
    observation_times_ok = all(parse_time(row["start_time"]) <= parse_time(row["end_time"]) for row in tables["observations"])
    checks["timestamps_are_ordered"] = trace_times_ok and observation_times_ok
    checks["trace_duration_matches_timestamps"] = all(
        abs((parse_time(row["end_time"]) - parse_time(row["start_time"])).total_seconds() * 1000 - row["duration_ms"]) <= 1.1
        for row in tables["traces"]
    )

    trace_tool_counts = Counter(row["trace_id"] for row in tables["tool_calls"])
    trace_generation_counts = Counter(row["trace_id"] for row in tables["generations"])
    checks["trace_metadata_counts_match"] = all(
        row["metadata"]["tool_count"] == trace_tool_counts[row["trace_id"]]
        and row["metadata"]["llm_rounds"] == trace_generation_counts[row["trace_id"]]
        for row in tables["traces"]
    )

    attempts_by_session: dict[str, list[int]] = defaultdict(list)
    for row in tables["traces"]:
        attempts_by_session[row["session_id"]].append(row["attempt_number"])
    checks["attempt_numbers_are_contiguous"] = all(
        sorted(values) == list(range(1, len(values) + 1)) for values in attempts_by_session.values()
    )

    statuses = Counter(row["status"] for row in tables["traces"])
    tool_statuses = Counter(row["status"] for row in tables["tool_calls"])
    difficulties = Counter(row["difficulty"] for row in tables["tasks"])
    task_model_pairs = {(row["task_id"], row["model_id"]) for row in tables["model_answers"]}
    checks["has_success_and_error_traces"] = statuses["success"] > 0 and statuses["error"] > 0
    checks["has_success_and_error_tools"] = tool_statuses["success"] > 0 and tool_statuses["error"] > 0
    checks["has_easy_and_hard_tasks"] = difficulties["easy"] > 0 and difficulties["hard"] > 0
    checks["every_task_has_every_model"] = all((task_id, model_id) in task_model_pairs for task_id in task_ids for model_id in model_ids)
    checks["has_retries"] = any(len(values) > 1 for values in attempts_by_session.values())
    checks["has_shared_annotations"] = len(tables["annotations"]) >= len(task_ids) * 2
    checks["has_annotation_revisions"] = any(row.get("supersedes_annotation_id") for row in tables["annotations"])
    checks["data_marked_synthetic"] = all(row.get("metadata", {}).get("synthetic") is True for row in tables["traces"])

    details.update({
        "counts": {name: len(rows) for name, rows in tables.items()},
        "trace_status": dict(sorted(statuses.items())),
        "tool_status": dict(sorted(tool_statuses.items())),
        "difficulty": dict(sorted(difficulties.items())),
        "sessions_with_retries": sum(len(values) > 1 for values in attempts_by_session.values()),
        "annotation_revisions": sum(bool(row.get("supersedes_annotation_id")) for row in tables["annotations"]),
    })
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"passed": not failed, "failed_checks": failed, "checks": checks, "details": details}


def main() -> None:
    args = parse_args()
    report = validate(args.data_dir.expanduser().resolve())
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
