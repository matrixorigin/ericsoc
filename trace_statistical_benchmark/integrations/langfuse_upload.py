#!/usr/bin/env python3
"""Upload a canonical trace corpus to Langfuse with resume support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from langfuse import Langfuse
from langfuse.api.commons.types import ObservationLevel
from langfuse.api.ingestion import (
    CreateGenerationBody,
    CreateSpanBody,
    IngestionEvent_GenerationCreate,
    IngestionEvent_SpanCreate,
    IngestionEvent_TraceCreate,
    TraceBody,
)


DEFAULT_DATA_DIR = Path("benchmark_data/simulation/canonical_data")
DEFAULT_ENVIRONMENT = "synthetic-simulation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--state-file", type=Path, default=Path("output/upload_state.json"))
    parser.add_argument("--report", type=Path, default=Path("output/upload_report.json"))
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--trace-limit", type=int, default=0, help="0 uploads every trace.")
    parser.add_argument("--batch-event-limit", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--send", action="store_true")
    parser.add_argument(
        "--restart-state",
        action="store_true",
        help="Start a new local progress record. Stable event IDs still prevent duplicate ingestion.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def event_id(dataset_version: str, kind: str, source_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset_version}:{kind}:{source_id}"))


def envelope_time() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def observation_level(status: str) -> ObservationLevel:
    return ObservationLevel.ERROR if status == "error" else ObservationLevel.DEFAULT


def clean_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    cleaned = {str(key): int(item) for key, item in value.items() if item is not None}
    return cleaned or None


def build_events(
    trace: dict[str, Any],
    observations: Iterable[dict[str, Any]],
    environment: str,
    dataset_version: str,
) -> list[Any]:
    trace_id = str(trace["trace_id"])
    metadata = {
        **(trace.get("metadata") or {}),
        "source": trace.get("source"),
        "status": trace.get("status"),
        "duration_ms": trace.get("duration_ms"),
        "cost": trace.get("cost"),
        "model": trace.get("model"),
        "model_id": trace.get("model_id"),
        "task_id": trace.get("task_id"),
        "attempt_number": trace.get("attempt_number"),
        "usage": trace.get("usage"),
        "dataset_version": dataset_version,
    }
    events: list[Any] = [
        IngestionEvent_TraceCreate(
            id=event_id(dataset_version, "trace", trace_id),
            timestamp=envelope_time(),
            body=TraceBody(
                id=trace_id,
                timestamp=parse_time(trace["start_time"]),
                name=str(trace["name"]),
                input=trace.get("input"),
                output=trace.get("output") or trace.get("error"),
                session_id=str(trace["session_id"]),
                version=dataset_version,
                release="trace-benchmark-v3",
                metadata=metadata,
                tags=["synthetic", "simulation", "trace-benchmark-v3", "full-roundtrip"],
                environment=environment,
            ),
        )
    ]
    for observation in observations:
        observation_id = str(observation["observation_id"])
        common = {
            "trace_id": trace_id,
            "id": observation_id,
            "name": str(observation["name"]),
            "start_time": parse_time(observation["start_time"]),
            "end_time": parse_time(observation["end_time"]),
            "parent_observation_id": observation.get("parent_observation_id"),
            "input": observation.get("input"),
            "output": observation.get("output"),
            "metadata": {
                **(observation.get("metadata") or {}),
                "canonical_type": observation.get("type"),
                "canonical_status": observation.get("status"),
                "dataset_version": dataset_version,
            },
            "level": observation_level(str(observation.get("status"))),
            "status_message": "Imported synthetic error" if observation.get("status") == "error" else None,
            "version": dataset_version,
            "environment": environment,
        }
        if observation.get("type") == "generation":
            events.append(
                IngestionEvent_GenerationCreate(
                    id=event_id(dataset_version, "generation", observation_id),
                    timestamp=envelope_time(),
                    body=CreateGenerationBody(
                        **common,
                        model=observation.get("model"),
                        usage_details=clean_usage(observation.get("usage")),
                    ),
                )
            )
        else:
            events.append(
                IngestionEvent_SpanCreate(
                    id=event_id(dataset_version, "span", observation_id),
                    timestamp=envelope_time(),
                    body=CreateSpanBody(**common),
                )
            )
    return events


def require_environment() -> tuple[str, str, str]:
    values = {
        "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY"),
        "LANGFUSE_BASE_URL": os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing))
    return (
        str(values["LANGFUSE_PUBLIC_KEY"]),
        str(values["LANGFUSE_SECRET_KEY"]),
        str(values["LANGFUSE_BASE_URL"]).rstrip("/"),
    )


def load_source(data_dir: Path, trace_limit: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    traces = read_jsonl(data_dir / "traces.jsonl")
    if trace_limit > 0:
        traces = traces[:trace_limit]
    selected_ids = {str(row["trace_id"]) for row in traces}
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(data_dir / "observations.jsonl"):
        trace_id = str(row["trace_id"])
        if trace_id in selected_ids:
            by_trace[trace_id].append(row)
    for rows in by_trace.values():
        rows.sort(key=lambda row: (row["start_time"], row["observation_id"]))
    return traces, by_trace, manifest


def iter_batches(
    traces: Iterable[dict[str, Any]],
    observations_by_trace: dict[str, list[dict[str, Any]]],
    completed: set[str],
    event_limit: int,
    environment: str,
    dataset_version: str,
) -> Iterator[tuple[list[str], list[Any]]]:
    trace_ids: list[str] = []
    events: list[Any] = []
    for trace in traces:
        trace_id = str(trace["trace_id"])
        if trace_id in completed:
            continue
        unit = build_events(trace, observations_by_trace.get(trace_id, []), environment, dataset_version)
        if events and len(events) + len(unit) > event_limit:
            yield trace_ids, events
            trace_ids, events = [], []
        trace_ids.append(trace_id)
        events.extend(unit)
    if events:
        yield trace_ids, events


def load_state(path: Path, dataset_version: str, source_sha256: str, restart: bool) -> dict[str, Any]:
    if path.exists() and not restart:
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("dataset_version") != dataset_version or state.get("source_sha256") != source_sha256:
            raise SystemExit("Upload state belongs to different source data. Use --restart-state after checking the paths.")
        return state
    return {
        "upload_version": "full-langfuse-roundtrip-v1",
        "dataset_version": dataset_version,
        "source_sha256": source_sha256,
        "completed_trace_ids": [],
        "completed_events": 0,
        "batch_count": 0,
        "started_at": envelope_time(),
        "updated_at": envelope_time(),
    }


def send_with_retry(client: Langfuse, events: list[Any], max_retries: int) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.api.ingestion.batch(
                batch=events,
                metadata={"source": "synthetic-simulation-corpus", "adapter_version": "1.0.0"},
            )
            errors = getattr(response, "errors", None) or []
            if errors:
                raise RuntimeError(f"Langfuse rejected part of the batch: {errors}")
            return
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 10))


def main() -> None:
    args = parse_args()
    if args.batch_event_limit < 1 or args.max_retries < 1 or args.trace_limit < 0:
        raise SystemExit("Batch limit and retries must be positive; trace limit cannot be negative.")

    traces_path = args.data_dir / "traces.jsonl"
    traces, observations_by_trace, manifest = load_source(args.data_dir, args.trace_limit)
    dataset_version = str(manifest["dataset_version"])
    source_sha256 = sha256_file(traces_path)
    observation_count = sum(len(rows) for rows in observations_by_trace.values())
    oversized_units = sum(1 + len(observations_by_trace.get(str(row["trace_id"]), [])) > args.batch_event_limit for row in traces)
    preview = {
        "mode": "send" if args.send else "dry-run",
        "dataset_version": dataset_version,
        "environment": args.environment,
        "trace_count": len(traces),
        "observation_count": observation_count,
        "ingestion_event_count": len(traces) + observation_count,
        "batch_event_limit": args.batch_event_limit,
        "oversized_trace_units": oversized_units,
        "source_sha256": source_sha256,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if not args.send:
        print("Dry run only. Add --send to start or resume the upload.")
        return

    public_key, secret_key, base_url = require_environment()
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        tracing_enabled=False,
        timeout=90,
    )
    if not client.auth_check():
        raise SystemExit("Langfuse authentication failed. Check the URL and API keys.")

    state = load_state(args.state_file, dataset_version, source_sha256, args.restart_state)
    all_completed = set(str(value) for value in state.get("completed_trace_ids", []))
    target_ids = {str(row["trace_id"]) for row in traces}
    completed = all_completed & target_ids
    pending_count = len(target_ids - completed)
    print(f"Resuming with {len(completed):,} completed and {pending_count:,} pending traces.")

    for batch_trace_ids, events in iter_batches(
        traces,
        observations_by_trace,
        completed,
        args.batch_event_limit,
        args.environment,
        dataset_version,
    ):
        send_with_retry(client, events, args.max_retries)
        completed.update(batch_trace_ids)
        all_completed.update(batch_trace_ids)
        state["completed_trace_ids"] = sorted(all_completed)
        state["completed_events"] = int(state.get("completed_events", 0)) + len(events)
        state["batch_count"] = int(state.get("batch_count", 0)) + 1
        state["updated_at"] = envelope_time()
        write_json(args.state_file, state)
        if state["batch_count"] % 25 == 0 or len(completed) == len(target_ids):
            print(
                f"Uploaded {len(completed):,}/{len(target_ids):,} traces "
                f"in {state['batch_count']:,} batches."
            )

    report = {
        **preview,
        "mode": "send",
        "completed_trace_count": len(completed),
        "all_selected_traces_completed": completed == target_ids,
        "batch_count": state["batch_count"],
        "finished_at": envelope_time(),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
