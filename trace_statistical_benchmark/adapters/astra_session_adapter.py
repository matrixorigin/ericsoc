#!/usr/bin/env python3
"""Convert Astra native session logs into the benchmark canonical tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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

SECRET_PATTERNS = [
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)(api[_ -]?key|secret[_ -]?key|password)(\s*[:=]\s*)([^\s\"']+)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astra-home", type=Path, default=Path.home() / ".astra")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", action="append", dest="session_ids")
    parser.add_argument("--max-traces", type=int, default=10)
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_hex(*parts: Any) -> str:
    raw = "\x1f".join(json.dumps(part, ensure_ascii=False, sort_keys=True) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def iso_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_text(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str):
        return value
    cleaned = value
    for pattern in SECRET_PATTERNS:
        if pattern.groups == 3:
            cleaned = pattern.sub(r"\1\2[REDACTED]", cleaned)
        else:
            cleaned = pattern.sub("[REDACTED]", cleaned)
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + f"\n...[truncated {len(cleaned) - max_chars} chars]"
    return cleaned


def normal_status(value: dict[str, Any]) -> str:
    return "error" if value.get("type") == "turn_error" else "success"


def find_session_files(astra_home: Path, session_ids: list[str] | None) -> list[Path]:
    session_dir = astra_home / "sessions"
    if session_ids:
        files = [session_dir / f"{session_id}.jsonl" for session_id in session_ids]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise SystemExit("Missing Astra session logs: " + ", ".join(missing))
        return files
    return sorted(path for path in session_dir.glob("*.jsonl") if path.is_file())


def event_in_attempt(event: dict[str, Any], attempt: dict[str, Any]) -> bool:
    if event.get("turn") != attempt["record"].get("turn"):
        return False
    event_time = iso_time(event.get("ts"))
    return attempt["start"] <= event_time <= attempt["end"] + timedelta(seconds=1)


def convert(astra_home: Path, session_files: list[Path], max_traces: int, max_chars: int) -> dict[str, list[dict[str, Any]]]:
    bundle: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_FILES}
    models: dict[str, dict[str, Any]] = {}
    tasks: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    session_rows: dict[str, dict[str, Any]] = {}

    for session_file in session_files:
        events = read_jsonl(session_file)
        if not events:
            continue
        session_id = str(events[0].get("session_id") or session_file.stem)
        attempt_counter: defaultdict[int, int] = defaultdict(int)
        turn_records = [row for row in events if row.get("type") in {"turn", "turn_error"}]

        for record in turn_records:
            turn = int(record.get("turn") or 1)
            attempt_counter[turn] += 1
            end = iso_time(record.get("ts"))
            duration = max(float(record.get("duration_ms") or 0), 1.0)
            start = end - timedelta(milliseconds=duration)
            trace_id = stable_hex("astra-trace", session_id, turn, attempt_counter[turn], record.get("ts"))
            attempts.append({
                "record": record,
                "events": events,
                "session_id": session_id,
                "attempt_number": attempt_counter[turn],
                "trace_id": trace_id,
                "start": start,
                "end": end,
                "source_file": str(session_file),
            })

    attempts.sort(key=lambda item: item["start"])
    if max_traces > 0:
        attempts = attempts[:max_traces]

    for attempt in attempts:
        record = attempt["record"]
        session_id = attempt["session_id"]
        trace_id = attempt["trace_id"]
        turn = int(record.get("turn") or 1)
        model_name = str(record.get("model") or "astra-unknown-model")
        model_id = stable_hex("astra-model", model_name)
        task_id = stable_hex("astra-task", clean_text(record.get("user_input") or "", max_chars))
        status = normal_status(record)

        models.setdefault(model_id, {
            "model_id": model_id,
            "provider": "astra",
            "name": model_name,
            "version": None,
            "configuration": {},
        })
        tasks.setdefault(task_id, {
            "task_id": task_id,
            "task_family": "astra-user-turn",
            "difficulty": "unknown",
            "prompt_version": "astra-native-v1",
            "prompt_text": clean_text(record.get("user_input"), max_chars),
            "reference_outcome": None,
            "mapping_status": "one-attempt-per-astra-turn-record",
            "metadata": {"source_session_id": session_id, "turn": turn},
        })

        error = None
        if status == "error":
            error = {
                "type": "astra_turn_error",
                "message": clean_text(record.get("error") or "Unknown Astra error", max_chars),
                "retryable": bool((record.get("metadata") or {}).get("retryable")),
            }

        trace_row = {
            "trace_id": trace_id,
            "session_id": session_id,
            "task_id": task_id,
            "model_id": model_id,
            "model": model_name,
            "name": f"astra-turn-{turn}-attempt-{attempt['attempt_number']}",
            "attempt_number": attempt["attempt_number"],
            "status": status,
            "start_time": format_time(attempt["start"]),
            "end_time": format_time(attempt["end"]),
            "duration_ms": float(record.get("duration_ms") or 0),
            "usage": {
                "input": record.get("tokens_in"),
                "output": record.get("tokens_out"),
                "cache_read": record.get("cache_read_tokens"),
            },
            "cost": record.get("cost_usd") or record.get("cumulative_cost_usd"),
            "input": clean_text(record.get("user_input"), max_chars),
            "output": clean_text(record.get("assistant_output"), max_chars),
            "error": error,
            "source": "astra-native-session-log",
            "metadata": {
                "turn": turn,
                "tool_count": record.get("tool_count"),
                "llm_rounds": record.get("llm_rounds"),
                "native_run_id": (record.get("metadata") or {}).get("run_id"),
                "source_file": attempt["source_file"],
                "adapter_version": "1.0.0",
            },
        }
        bundle["traces"].append(trace_row)

        root_id = stable_hex("astra-observation-root", trace_id)
        bundle["observations"].append({
            "observation_id": root_id,
            "trace_id": trace_id,
            "parent_observation_id": None,
            "type": "span",
            "name": "astra-turn",
            "status": status,
            "start_time": trace_row["start_time"],
            "end_time": trace_row["end_time"],
            "input": trace_row["input"],
            "output": trace_row["output"] if status == "success" else error,
            "metadata": {"turn": turn, "attempt_number": attempt["attempt_number"]},
        })

        round_events = [row for row in attempt["events"] if row.get("type") == "llm_round" and event_in_attempt(row, attempt)]
        round_events.sort(key=lambda row: iso_time(row.get("ts")))
        for round_index, round_event in enumerate(round_events, 1):
            round_end = min(iso_time(round_event.get("ts")), attempt["end"])
            round_start = max(
                attempt["start"],
                round_end - timedelta(milliseconds=float(round_event.get("duration_ms") or 0)),
            )
            generation_id = stable_hex("astra-generation", trace_id, round_index)
            generation_usage = {
                "input": round_event.get("tokens_in"),
                "output": round_event.get("tokens_out"),
                "cache_read": round_event.get("cache_read_tokens"),
            }
            generation_output = {
                "finish_reason": (round_event.get("metadata") or {}).get("finish_reason"),
                "tool_call_names": (round_event.get("metadata") or {}).get("tool_call_names") or [],
            }
            bundle["observations"].append({
                "observation_id": generation_id,
                "trace_id": trace_id,
                "parent_observation_id": root_id,
                "type": "generation",
                "name": f"llm-round-{round_index}",
                "status": "success",
                "start_time": format_time(round_start),
                "end_time": format_time(round_end),
                "input": {"turn": turn, "agentic_step": round_event.get("agentic_step")},
                "output": generation_output,
                "model": model_name,
                "usage": generation_usage,
                "metadata": {
                    "native_run_id": (round_event.get("metadata") or {}).get("run_id"),
                    "round": round_event.get("round"),
                    "agentic_step": round_event.get("agentic_step"),
                },
            })
            bundle["generations"].append({
                "observation_id": generation_id,
                "trace_id": trace_id,
                "model_id": model_id,
                "input": {"turn": turn, "agentic_step": round_event.get("agentic_step")},
                "output": generation_output,
                "usage": generation_usage,
                "duration_ms": max(0.0, (round_end - round_start).total_seconds() * 1000),
                "metadata": {"round_index": round_index},
            })

            for tool_index, tool in enumerate(round_event.get("tool_calls") or [], 1):
                tool_start = round_start
                if tool.get("start_offset_ms") is not None:
                    tool_start = min(
                        attempt["end"],
                        attempt["start"] + timedelta(milliseconds=float(tool["start_offset_ms"])),
                    )
                tool_end = min(attempt["end"], tool_start + timedelta(milliseconds=float(tool.get("ms") or 0)))
                tool_id = stable_hex("astra-tool", trace_id, round_index, tool_index, tool.get("tool_call_id"))
                tool_status = "success" if tool.get("ok") is not False else "error"
                tool_input = clean_text(tool.get("args_preview") or tool.get("args_full"), max_chars)
                tool_output = clean_text(tool.get("result_preview") or tool.get("result_full"), max_chars)
                tool_metadata = {
                    "native_tool_call_id": tool.get("tool_call_id"),
                    "input_bytes": tool.get("input_bytes"),
                    "output_bytes": tool.get("output_bytes"),
                    "parallel": tool.get("parallel"),
                    "batch_id": tool.get("batch_id"),
                }
                bundle["observations"].append({
                    "observation_id": tool_id,
                    "trace_id": trace_id,
                    "parent_observation_id": generation_id,
                    "type": "tool",
                    "name": str(tool.get("name") or "unknown-tool"),
                    "status": tool_status,
                    "start_time": format_time(tool_start),
                    "end_time": format_time(tool_end),
                    "input": tool_input,
                    "output": tool_output,
                    "metadata": tool_metadata,
                })
                bundle["tool_calls"].append({
                    "observation_id": tool_id,
                    "trace_id": trace_id,
                    "tool_name": str(tool.get("name") or "unknown-tool"),
                    "tool_index": tool_index,
                    "status": tool_status,
                    "input": tool_input,
                    "output": tool_output,
                    "duration_ms": float(tool.get("ms") or 0),
                    "error": None if tool_status == "success" else tool_output,
                    "metadata": tool_metadata,
                })

        bundle["model_answers"].append({
            "answer_id": stable_hex("astra-answer", trace_id),
            "task_id": task_id,
            "trace_id": trace_id,
            "model_id": model_id,
            "answer_text": trace_row["output"],
            "parsed_answer": None,
            "correct": None,
            "grading_version": None,
            "metadata": {"source": "astra-turn-record"},
        })

        current_session = session_rows.get(session_id)
        if current_session is None:
            session_rows[session_id] = {
                "session_id": session_id,
                "source": "astra-native-session-log",
                "start_time": trace_row["start_time"],
                "end_time": trace_row["end_time"],
                "metadata": {"trace_count": 1},
            }
        else:
            current_session["start_time"] = min(current_session["start_time"], trace_row["start_time"])
            current_session["end_time"] = max(current_session["end_time"], trace_row["end_time"])
            current_session["metadata"]["trace_count"] += 1

    bundle["models"] = sorted(models.values(), key=lambda row: row["model_id"])
    bundle["tasks"] = sorted(tasks.values(), key=lambda row: row["task_id"])
    bundle["sessions"] = sorted(session_rows.values(), key=lambda row: row["session_id"])
    return bundle


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.replace:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Add --replace to overwrite it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    session_files = find_session_files(args.astra_home.expanduser(), args.session_ids)
    bundle = convert(args.astra_home.expanduser(), session_files, args.max_traces, args.max_text_chars)
    for table, filename in TABLE_FILES.items():
        write_jsonl(output_dir / filename, bundle[table])

    manifest = {
        "schema_version": "2.0.0",
        "adapter": "astra-native-session-log-v1",
        "source_files": [str(path) for path in session_files],
        "counts": {table: len(rows) for table, rows in bundle.items()},
        "notes": [
            "Astra sessions/*.jsonl is the source of truth.",
            "Encrypted step_events.jsonl files are intentionally ignored.",
            "Tool payloads use previews and redact common secret patterns.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
