#!/usr/bin/env python3
"""Convert Codex rollout JSONL files into the benchmark canonical tables."""

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
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-turn-id", action="append")
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
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
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def parse_time(value: str) -> datetime:
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
        cleaned = pattern.sub(r"\1\2[REDACTED]" if pattern.groups == 3 else "[REDACTED]", cleaned)
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + f"\n...[truncated {len(cleaned) - max_chars} chars]"
    return cleaned


def message_text(payload: dict[str, Any]) -> str:
    values: list[str] = []
    for item in payload.get("content") or []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            values.append(item["text"])
    return "\n".join(values)


def output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(item.get("text"))
            for item in value
            if isinstance(item, dict) and item.get("text") is not None
        )
    return json.dumps(value, ensure_ascii=False) if value is not None else ""


def split_turns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in rows:
        payload = row.get("payload") or {}
        if row.get("type") == "event_msg" and payload.get("type") == "task_started":
            if current is not None:
                turns.append(current)
            current = {
                "turn_id": payload.get("turn_id"),
                "start": parse_time(row["timestamp"]),
                "end": None,
                "complete": False,
                "events": [row],
            }
            continue
        if current is None:
            continue
        current["events"].append(row)
        if row.get("type") == "event_msg" and payload.get("type") == "task_complete":
            current["end"] = parse_time(row["timestamp"])
            current["complete"] = True
            turns.append(current)
            current = None
    if current is not None:
        turns.append(current)
    return turns


def convert_rollouts(
    rollout_files: list[Path], include_turn_ids: set[str] | None = None, max_chars: int = 12000
) -> dict[str, list[dict[str, Any]]]:
    bundle: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_FILES}
    models: dict[str, dict[str, Any]] = {}
    tasks: dict[str, dict[str, Any]] = {}
    sessions: dict[str, dict[str, Any]] = {}

    for rollout_file in rollout_files:
        rows = read_jsonl(rollout_file)
        meta = next((row.get("payload") or {} for row in rows if row.get("type") == "session_meta"), {})
        if meta.get("thread_source") == "subagent" or isinstance(meta.get("source"), dict):
            raise ValueError(f"Subagent rollout is not a primary answer log: {rollout_file}")
        session_id = str(meta.get("id") or rollout_file.stem)
        all_turns = split_turns(rows)
        turns = [turn for turn in all_turns if not include_turn_ids or turn["turn_id"] in include_turn_ids]

        for turn in turns:
            events = turn["events"]
            context = next(
                (
                    row.get("payload") or {}
                    for row in events
                    if row.get("type") == "turn_context"
                    and (row.get("payload") or {}).get("turn_id") == turn["turn_id"]
                ),
                {},
            )
            model_name = str(context.get("model") or meta.get("model_provider") or "codex-unknown-model")
            model_id = stable_hex("codex-model", model_name)
            start = turn["start"]
            end = turn["end"] or parse_time(events[-1]["timestamp"])

            user_messages = []
            assistant_messages = []
            for row in events:
                payload = row.get("payload") or {}
                if row.get("type") != "response_item" or payload.get("type") != "message":
                    continue
                text = message_text(payload)
                if payload.get("role") == "user" and not text.lstrip().startswith("<recommended_plugins>"):
                    user_messages.append(text)
                elif payload.get("role") == "assistant" and text.strip():
                    assistant_messages.append(text)
            trace_input = clean_text("\n\n".join(user_messages), max_chars)
            trace_output = clean_text(assistant_messages[-1] if assistant_messages else "", max_chars)
            status = "success" if turn["complete"] else "error"
            trace_id = stable_hex("codex-trace", session_id, turn["turn_id"])
            source_task_id = stable_hex("codex-task", trace_input)

            token_events = [
                row for row in events
                if row.get("type") == "event_msg" and (row.get("payload") or {}).get("type") == "token_count"
            ]
            usages = [((row.get("payload") or {}).get("info") or {}).get("last_token_usage") or {} for row in token_events]
            usage = {
                "input": sum(int(item.get("input_tokens") or 0) for item in usages),
                "output": sum(int(item.get("output_tokens") or 0) for item in usages),
                "cache_read": sum(int(item.get("cached_input_tokens") or 0) for item in usages),
            }
            call_rows = [
                row for row in events
                if row.get("type") == "response_item"
                and (row.get("payload") or {}).get("type") in {"custom_tool_call", "function_call"}
            ]

            models.setdefault(model_id, {
                "model_id": model_id,
                "provider": "openai-codex",
                "name": model_name,
                "version": None,
                "configuration": {"reasoning_effort": context.get("effort")},
            })
            tasks.setdefault(source_task_id, {
                "task_id": source_task_id,
                "task_family": "codex-user-turn",
                "difficulty": "unknown",
                "prompt_version": "codex-rollout-v1",
                "prompt_text": trace_input,
                "reference_outcome": None,
                "mapping_status": "one-task-started-event-per-trace",
                "metadata": {"turn_id": turn["turn_id"]},
            })
            trace_row = {
                "trace_id": trace_id,
                "session_id": session_id,
                "task_id": source_task_id,
                "model_id": model_id,
                "model": model_name,
                "name": f"codex-turn-{turn['turn_id']}",
                "attempt_number": 1,
                "status": status,
                "start_time": format_time(start),
                "end_time": format_time(end),
                "duration_ms": (end - start).total_seconds() * 1000,
                "usage": usage,
                "cost": None,
                "input": trace_input,
                "output": trace_output,
                "error": None if status == "success" else {"type": "incomplete_turn", "message": "No task_complete event"},
                "source": "codex-rollout-jsonl",
                "metadata": {
                    "turn_id": turn["turn_id"],
                    "turn": None,
                    "tool_count": len(call_rows),
                    "llm_rounds": len(token_events),
                    "source_file": str(rollout_file),
                    "adapter_version": "1.0.0",
                },
            }
            bundle["traces"].append(trace_row)

            root_id = stable_hex("codex-root", trace_id)
            bundle["observations"].append({
                "observation_id": root_id,
                "trace_id": trace_id,
                "parent_observation_id": None,
                "type": "span",
                "name": "codex-turn",
                "status": status,
                "start_time": trace_row["start_time"],
                "end_time": trace_row["end_time"],
                "input": trace_input,
                "output": trace_output,
                "metadata": {"turn_id": turn["turn_id"]},
            })

            previous_boundary = start
            for index, token_row in enumerate(token_events, 1):
                generation_end = parse_time(token_row["timestamp"])
                generation_start = min(previous_boundary, generation_end)
                if generation_start == generation_end:
                    generation_start -= timedelta(milliseconds=1)
                step_usage_raw = (((token_row.get("payload") or {}).get("info") or {}).get("last_token_usage") or {})
                step_usage = {
                    "input": step_usage_raw.get("input_tokens"),
                    "output": step_usage_raw.get("output_tokens"),
                    "cache_read": step_usage_raw.get("cached_input_tokens"),
                }
                generation_id = stable_hex("codex-generation", trace_id, index)
                generation = {
                    "observation_id": generation_id,
                    "trace_id": trace_id,
                    "parent_observation_id": root_id,
                    "type": "generation",
                    "name": f"agent-model-step-{index}",
                    "status": "success",
                    "start_time": format_time(generation_start),
                    "end_time": format_time(generation_end),
                    "input": {"turn_id": turn["turn_id"], "step": index},
                    "output": {"note": "Interval ends when Codex records token usage."},
                    "model": model_name,
                    "usage": step_usage,
                    "metadata": {"timing_semantics": "agent-step interval, not provider-only latency"},
                }
                bundle["observations"].append(generation)
                bundle["generations"].append({
                    "observation_id": generation_id,
                    "trace_id": trace_id,
                    "model_id": model_id,
                    "input": generation["input"],
                    "output": generation["output"],
                    "usage": step_usage,
                    "duration_ms": (generation_end - generation_start).total_seconds() * 1000,
                    "metadata": generation["metadata"],
                })
                previous_boundary = generation_end

            output_by_call: dict[str, dict[str, Any]] = {}
            for row in events:
                payload = row.get("payload") or {}
                if row.get("type") == "response_item" and payload.get("type") in {"custom_tool_call_output", "function_call_output"}:
                    output_by_call[str(payload.get("call_id"))] = row
            for index, call_row in enumerate(call_rows, 1):
                payload = call_row.get("payload") or {}
                call_id = str(payload.get("call_id") or payload.get("id") or index)
                result_row = output_by_call.get(call_id)
                tool_start = parse_time(call_row["timestamp"])
                tool_end = parse_time(result_row["timestamp"]) if result_row else tool_start + timedelta(milliseconds=1)
                result_payload = (result_row or {}).get("payload") or {}
                result_text = output_text(result_payload.get("output"))
                lowered = result_text.lower()
                tool_status = "error" if any(marker in lowered for marker in ("process exited with code 1", "iserror\":true", "script failed")) else "success"
                tool_name = str(payload.get("name") or "unknown-tool")
                tool_id = stable_hex("codex-tool", trace_id, call_id)
                tool_input = clean_text(payload.get("input") or payload.get("arguments") or "", max_chars)
                tool_output = clean_text(result_text, max_chars)
                tool_metadata = {"native_call_id": call_id, "tool_index": index}
                bundle["observations"].append({
                    "observation_id": tool_id,
                    "trace_id": trace_id,
                    "parent_observation_id": root_id,
                    "type": "tool",
                    "name": tool_name,
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
                    "tool_name": tool_name,
                    "tool_index": index,
                    "status": tool_status,
                    "input": tool_input,
                    "output": tool_output,
                    "duration_ms": (tool_end - tool_start).total_seconds() * 1000,
                    "error": tool_output if tool_status == "error" else None,
                    "metadata": tool_metadata,
                })

            bundle["model_answers"].append({
                "answer_id": stable_hex("codex-answer", trace_id),
                "task_id": source_task_id,
                "trace_id": trace_id,
                "model_id": model_id,
                "answer_text": trace_output,
                "parsed_answer": None,
                "correct": None,
                "grading_version": None,
                "metadata": {"source": "codex-rollout-jsonl"},
            })

            session = sessions.setdefault(session_id, {
                "session_id": session_id,
                "source": "codex-rollout-jsonl",
                "start_time": trace_row["start_time"],
                "end_time": trace_row["end_time"],
                "metadata": {"trace_count": 0, "source_file": str(rollout_file)},
            })
            session["start_time"] = min(session["start_time"], trace_row["start_time"])
            session["end_time"] = max(session["end_time"], trace_row["end_time"])
            session["metadata"]["trace_count"] += 1

    bundle["models"] = sorted(models.values(), key=lambda row: row["model_id"])
    bundle["tasks"] = sorted(tasks.values(), key=lambda row: row["task_id"])
    bundle["sessions"] = sorted(sessions.values(), key=lambda row: row["session_id"])
    return bundle


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.replace:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Add --replace to overwrite it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_files = [path.expanduser().resolve() for path in args.rollout]
    bundle = convert_rollouts(rollout_files, set(args.include_turn_id or []), args.max_text_chars)
    for table, filename in TABLE_FILES.items():
        write_jsonl(output_dir / filename, bundle[table])
    manifest = {
        "schema_version": "2.0.0",
        "adapter": "codex-rollout-jsonl-v1",
        "source_files": [str(path) for path in rollout_files],
        "counts": {table: len(rows) for table, rows in bundle.items()},
        "notes": [
            "Primary user rollouts only; subagent rollouts are rejected.",
            "Generation timing is an agent-step interval, not provider-only model latency.",
            "Common secret patterns are redacted and long text is truncated.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
