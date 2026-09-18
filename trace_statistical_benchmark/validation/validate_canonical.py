"""Validate canonical v2 tables and annotation history. / 检查 v2 表和标注历史。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tools.canonical_v2 import load_bundle, parse_time
except ModuleNotFoundError:  # Direct execution from tools/.
    from canonical_v2 import load_bundle, parse_time  # type: ignore


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

REQUIRED_FIELDS = {
    "models": {"model_id", "name", "configuration"},
    "tasks": {"task_id", "task_family", "difficulty", "prompt_version", "mapping_status"},
    "sessions": {"session_id", "source", "start_time", "end_time", "metadata"},
    "traces": {"trace_id", "session_id", "task_id", "model_id", "attempt_number", "status", "start_time", "end_time", "duration_ms", "usage", "metadata"},
    "observations": {"observation_id", "trace_id", "parent_observation_id", "type", "name", "status", "start_time", "end_time", "metadata"},
    "generations": {"observation_id", "trace_id", "model_id", "usage", "duration_ms", "metadata"},
    "tool_calls": {"observation_id", "trace_id", "tool_name", "tool_index", "status", "duration_ms", "metadata"},
    "model_answers": {"answer_id", "task_id", "trace_id", "model_id", "answer_text", "correct", "grading_version", "metadata"},
    "annotations": {"annotation_id", "target_type", "target_id", "trace_id", "label_schema_version", "label_json", "annotator_id", "annotator_type", "labeled_at", "ingested_at", "source", "supersedes_annotation_id", "retracted", "metadata"},
}


def _indexes(bundle: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        table: {str(row.get(PRIMARY_KEYS[table])): row for row in rows}
        for table, rows in bundle.items()
    }


def validate_bundle(bundle: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    for table, rows in bundle.items():
        key = PRIMARY_KEYS[table]
        keys = [row.get(key) for row in rows]
        missing_keys = [index for index, value in enumerate(keys, 1) if not isinstance(value, str) or not value]
        if missing_keys:
            errors.append(f"{table}: missing {key} at rows {missing_keys}")
        duplicates = sorted(value for value, count in Counter(keys).items() if value is not None and count > 1)
        if duplicates:
            errors.append(f"{table}: duplicate {key}: {duplicates}")
        for index, row in enumerate(rows, 1):
            missing = sorted(REQUIRED_FIELDS[table] - row.keys())
            if missing:
                errors.append(f"{table} row {index}: missing fields {missing}")

    idx = _indexes(bundle)
    models = idx["models"]
    tasks = idx["tasks"]
    sessions = idx["sessions"]
    traces = idx["traces"]
    observations = idx["observations"]
    annotations = idx["annotations"]

    def require_fk(table: str, row_id: str, field: str, value: Any, target: dict[str, Any]) -> None:
        if value not in target:
            errors.append(f"{table} {row_id}: {field}={value!r} does not exist")

    for session in bundle["sessions"]:
        _check_time_range("session", session["session_id"], session, errors)

    for trace in bundle["traces"]:
        trace_id = trace["trace_id"]
        require_fk("trace", trace_id, "session_id", trace.get("session_id"), sessions)
        require_fk("trace", trace_id, "task_id", trace.get("task_id"), tasks)
        require_fk("trace", trace_id, "model_id", trace.get("model_id"), models)
        _check_time_range("trace", trace_id, trace, errors)
        if not isinstance(trace.get("attempt_number"), int) or trace["attempt_number"] < 1:
            errors.append(f"trace {trace_id}: attempt_number must be a positive integer")

    for observation in bundle["observations"]:
        observation_id = observation["observation_id"]
        require_fk("observation", observation_id, "trace_id", observation.get("trace_id"), traces)
        parent_id = observation.get("parent_observation_id")
        if parent_id is not None:
            require_fk("observation", observation_id, "parent_observation_id", parent_id, observations)
            parent = observations.get(parent_id)
            if parent and parent.get("trace_id") != observation.get("trace_id"):
                errors.append(f"observation {observation_id}: parent belongs to a different trace")
        if observation.get("type") not in {"span", "generation", "tool", "event"}:
            errors.append(f"observation {observation_id}: unsupported type {observation.get('type')!r}")
        _check_time_range("observation", observation_id, observation, errors)

    for generation in bundle["generations"]:
        observation_id = generation["observation_id"]
        require_fk("generation", observation_id, "observation_id", observation_id, observations)
        require_fk("generation", observation_id, "trace_id", generation.get("trace_id"), traces)
        require_fk("generation", observation_id, "model_id", generation.get("model_id"), models)
        source = observations.get(observation_id)
        if source and source.get("type") != "generation":
            errors.append(f"generation {observation_id}: linked observation is not a generation")
        if source and source.get("trace_id") != generation.get("trace_id"):
            errors.append(f"generation {observation_id}: trace_id disagrees with observation")

    for tool_call in bundle["tool_calls"]:
        observation_id = tool_call["observation_id"]
        require_fk("tool_call", observation_id, "observation_id", observation_id, observations)
        require_fk("tool_call", observation_id, "trace_id", tool_call.get("trace_id"), traces)
        source = observations.get(observation_id)
        if source and source.get("type") != "tool":
            errors.append(f"tool_call {observation_id}: linked observation is not a tool")
        if source and source.get("trace_id") != tool_call.get("trace_id"):
            errors.append(f"tool_call {observation_id}: trace_id disagrees with observation")

    for answer in bundle["model_answers"]:
        answer_id = answer["answer_id"]
        require_fk("model_answer", answer_id, "task_id", answer.get("task_id"), tasks)
        require_fk("model_answer", answer_id, "trace_id", answer.get("trace_id"), traces)
        require_fk("model_answer", answer_id, "model_id", answer.get("model_id"), models)

    allowed_annotator_types = {"human", "llm", "rule", "imported"}
    for annotation in bundle["annotations"]:
        annotation_id = annotation["annotation_id"]
        target_type = annotation.get("target_type")
        target_id = annotation.get("target_id")
        target_table = {"trace": traces, "observation": observations, "session": sessions}.get(target_type)
        if target_table is None:
            errors.append(f"annotation {annotation_id}: unsupported target_type {target_type!r}")
        elif target_id not in target_table:
            errors.append(f"annotation {annotation_id}: target {target_type}:{target_id} does not exist")

        linked_trace_id = annotation.get("trace_id")
        if target_type == "trace" and linked_trace_id != target_id:
            errors.append(f"annotation {annotation_id}: trace target must use the same trace_id")
        if target_type == "observation":
            source = observations.get(target_id)
            if source and linked_trace_id != source.get("trace_id"):
                errors.append(f"annotation {annotation_id}: trace_id disagrees with target observation")
        if target_type == "session" and linked_trace_id is not None:
            errors.append(f"annotation {annotation_id}: session target must not set trace_id")

        if annotation.get("annotator_type") not in allowed_annotator_types:
            errors.append(f"annotation {annotation_id}: unsupported annotator_type")
        if not isinstance(annotation.get("label_json"), dict):
            errors.append(f"annotation {annotation_id}: label_json must be an object")
        else:
            confidence = annotation["label_json"].get("confidence")
            if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
                errors.append(f"annotation {annotation_id}: confidence must be between 0 and 1")
        _check_annotation_times(annotation, errors)

        previous_id = annotation.get("supersedes_annotation_id")
        if previous_id is not None:
            previous = annotations.get(previous_id)
            if previous is None:
                errors.append(f"annotation {annotation_id}: superseded annotation {previous_id!r} does not exist")
            else:
                for field in ("target_type", "target_id", "annotator_id", "label_schema_version"):
                    if previous.get(field) != annotation.get(field):
                        errors.append(f"annotation {annotation_id}: revision changes {field}")
                try:
                    if parse_time(previous["labeled_at"]) >= parse_time(annotation["labeled_at"]):
                        errors.append(f"annotation {annotation_id}: revision is not later than its predecessor")
                except (KeyError, TypeError, ValueError):
                    pass

    _check_revision_cycles(annotations, errors)

    unresolved = sum(task.get("mapping_status") != "resolved" for task in bundle["tasks"])
    if unresolved:
        warnings.append(
            f"{unresolved} task rows are unresolved; matched cross-model comparisons must wait for source task mapping"
        )

    return {
        "valid": not errors,
        "schema_version": "2.0.0",
        "counts": {table: len(rows) for table, rows in bundle.items()},
        "errors": errors,
        "warnings": warnings,
    }


def _check_time_range(kind: str, row_id: str, row: dict[str, Any], errors: list[str]) -> None:
    try:
        start = parse_time(row["start_time"])
        end = parse_time(row["end_time"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{kind} {row_id}: invalid timestamp ({exc})")
        return
    if end < start:
        errors.append(f"{kind} {row_id}: end_time is before start_time")


def _check_annotation_times(annotation: dict[str, Any], errors: list[str]) -> None:
    annotation_id = annotation["annotation_id"]
    try:
        labeled_at = parse_time(annotation["labeled_at"])
        ingested_at = parse_time(annotation["ingested_at"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"annotation {annotation_id}: invalid timestamp ({exc})")
        return
    if ingested_at < labeled_at:
        errors.append(f"annotation {annotation_id}: ingested_at is before labeled_at")


def _check_revision_cycles(annotations: dict[str, dict[str, Any]], errors: list[str]) -> None:
    for annotation_id in annotations:
        seen: set[str] = set()
        current: str | None = annotation_id
        while current is not None and current in annotations:
            if current in seen:
                errors.append(f"annotation {annotation_id}: revision chain contains a cycle")
                break
            seen.add(current)
            current = annotations[current].get("supersedes_annotation_id")


def validate_dataset(data_dir: Path) -> dict[str, Any]:
    return validate_bundle(load_bundle(data_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--allow-invalid", action="store_true", help="Print errors without returning exit code 1")
    args = parser.parse_args()
    result = validate_dataset(args.data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"] and not args.allow_invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
