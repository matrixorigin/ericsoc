#!/usr/bin/env python3
"""Deterministic reference implementations for the baseline trace queries."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable


NOTEBOOK_PATTERN = re.compile(r"hourly_h3_analysis_v\d+\.ipynb", re.IGNORECASE)


TASKS: list[dict[str, Any]] = [
    {
        "id": "query01",
        "title": "Trace status totals",
        "question": "How many Astra notebook task attempts are in the imported scope, and how many ended in success or error?",
        "category": "basic",
        "difficulty": "easy",
    },
    {
        "id": "query02",
        "title": "Failed attempt details",
        "question": "Which imported task attempts failed? Return their trace ID, session ID, turn number, duration, error type, and tool count.",
        "category": "error-analysis",
        "difficulty": "easy",
    },
    {
        "id": "query03",
        "title": "Latest attempt per session",
        "question": "What is the latest recorded task attempt in each imported Astra session?",
        "category": "session-analysis",
        "difficulty": "medium",
    },
    {
        "id": "query04",
        "title": "Longest task attempt",
        "question": "Which task attempt took the longest, and what were its status, duration, tool count, and model-round count?",
        "category": "performance",
        "difficulty": "easy",
    },
    {
        "id": "query05",
        "title": "Tool usage summary",
        "question": "Across all imported traces, how many successful and failed calls were recorded for each tool name?",
        "category": "tool-analysis",
        "difficulty": "medium",
    },
    {
        "id": "query06",
        "title": "Failed tool calls",
        "question": "Which individual tool calls failed? Return the trace ID, tool name, tool index, and recorded duration.",
        "category": "error-analysis",
        "difficulty": "medium",
    },
    {
        "id": "query07",
        "title": "Token usage by trace",
        "question": "Report the recorded input, output, and cache-read tokens for each trace. Which trace has the largest input-plus-output total?",
        "category": "performance",
        "difficulty": "medium",
    },
    {
        "id": "query08",
        "title": "Generation count and P95 latency",
        "question": "For each trace, how many LLM generations were recorded and what is their nearest-rank P95 duration in milliseconds?",
        "category": "performance",
        "difficulty": "hard",
    },
    {
        "id": "query09",
        "title": "Slowest tool calls",
        "question": "What are the five slowest recorded tool calls across the imported traces?",
        "category": "performance",
        "difficulty": "medium",
    },
    {
        "id": "query10",
        "title": "Success versus error comparison",
        "question": "Compare successful and failed task attempts using average duration, tool count, input tokens, and output tokens. Do not claim statistical significance.",
        "category": "comparison",
        "difficulty": "hard",
    },
    {
        "id": "query11",
        "title": "Later success after failure",
        "question": "For each failed trace, was there a strictly later successful trace in the same session? If so, return the earliest one and the time gap.",
        "category": "recovery-analysis",
        "difficulty": "hard",
    },
    {
        "id": "query12",
        "title": "Session execution timeline",
        "question": "Build the ordered execution timeline for each session, including turn number, status, start time, duration, and trace ID.",
        "category": "session-analysis",
        "difficulty": "medium",
    },
    {
        "id": "query13",
        "title": "Final output availability",
        "question": "Which traces contain a nonblank final output? Report Unicode character length and summarize available versus missing outputs.",
        "category": "output-analysis",
        "difficulty": "easy",
    },
    {
        "id": "query14",
        "title": "Notebook versions referenced",
        "question": "Which hourly_h3_analysis notebook versions are mentioned in each trace's input, output, or tool previews?",
        "category": "artifact-analysis",
        "difficulty": "hard",
    },
    {
        "id": "query15",
        "title": "Canonical data quality audit",
        "question": "Audit the imported trace structure: counts, missing model/output/usage fields, root-span counts, parent links, and timestamp ordering.",
        "category": "data-quality",
        "difficulty": "hard",
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def duration_ms(row: dict[str, Any]) -> int:
    return round((parse_time(row["end_time"]) - parse_time(row["start_time"])).total_seconds() * 1000)


def rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def nearest_rank(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def load_data(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    traces = read_jsonl(data_dir / "traces.jsonl")
    observations = read_jsonl(data_dir / "observations.jsonl")
    traces.sort(key=lambda row: (row["start_time"], row["trace_id"]))
    observations.sort(key=lambda row: (row["start_time"], row["observation_id"]))
    return traces, observations


def query01(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["status"] for row in traces)
    return {"total_traces": len(traces), "success": counts["success"], "error": counts["error"], "other": len(traces) - counts["success"] - counts["error"]}


def query02(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for trace in traces:
        if trace["status"] != "error":
            continue
        rows.append({
            "trace_id": trace["trace_id"],
            "session_id": trace["session_id"],
            "turn": trace["metadata"].get("turn"),
            "duration_ms": trace["duration_ms"],
            "error_type": (trace.get("error") or {}).get("type"),
            "tool_count": trace["metadata"].get("tool_count"),
        })
    return {"rows": rows, "failed_trace_count": len(rows)}


def query03(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[trace["session_id"]].append(trace)
    rows = []
    for session_id, session_traces in sorted(grouped.items()):
        latest = max(session_traces, key=lambda row: (row["end_time"], row["trace_id"]))
        rows.append({"session_id": session_id, "trace_id": latest["trace_id"], "turn": latest["metadata"].get("turn"), "status": latest["status"], "end_time": latest["end_time"]})
    return {"rows": rows, "session_count": len(rows)}


def query04(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(row["duration_ms"] for row in traces)
    rows = [{
        "trace_id": row["trace_id"],
        "status": row["status"],
        "duration_ms": row["duration_ms"],
        "tool_count": row["metadata"].get("tool_count"),
        "llm_rounds": row["metadata"].get("llm_rounds"),
    } for row in traces if row["duration_ms"] == maximum]
    return {"rows": rows, "longest_duration_ms": maximum}


def query05(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in observations:
        if row["type"] == "tool":
            grouped[row["name"].removeprefix("tool:")][row["status"]] += 1
    rows = []
    for name, counts in sorted(grouped.items()):
        rows.append({"tool_name": name, "total": sum(counts.values()), "success": counts["success"], "error": counts["error"]})
    return {"rows": rows, "tool_call_count": sum(row["total"] for row in rows)}


def query06(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [{
        "trace_id": row["trace_id"],
        "tool_name": row["name"].removeprefix("tool:"),
        "tool_index": row["metadata"].get("tool_index"),
        "duration_ms": duration_ms(row),
    } for row in observations if row["type"] == "tool" and row["status"] == "error"]
    rows.sort(key=lambda row: (row["trace_id"], row["tool_index"]))
    return {"rows": rows, "failed_tool_call_count": len(rows)}


def query07(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for trace in traces:
        usage = trace.get("usage") or {}
        row = {
            "trace_id": trace["trace_id"],
            "input_tokens": int(usage.get("input") or 0),
            "output_tokens": int(usage.get("output") or 0),
            "cache_read_tokens": int(usage.get("cache_read") or 0),
        }
        row["input_plus_output_tokens"] = row["input_tokens"] + row["output_tokens"]
        rows.append(row)
    maximum = max(row["input_plus_output_tokens"] for row in rows)
    max_ids = sorted(row["trace_id"] for row in rows if row["input_plus_output_tokens"] == maximum)
    return {"rows": rows, "largest_input_plus_output": maximum, "largest_trace_ids": max_ids, "note": "Cache-read tokens are reported separately and are not added again."}


def query08(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in observations:
        if row["type"] == "generation":
            grouped[row["trace_id"]].append(duration_ms(row))
    rows = []
    for trace in traces:
        values = grouped.get(trace["trace_id"], [])
        rows.append({"trace_id": trace["trace_id"], "generation_count": len(values), "p95_duration_ms": nearest_rank(values, 0.95)})
    return {"rows": rows, "method": "nearest-rank: sorted_values[ceil(0.95*n)-1]"}


def query09(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    tools = [row for row in observations if row["type"] == "tool"]
    tools.sort(key=lambda row: (-duration_ms(row), row["trace_id"], row["observation_id"]))
    rows = [{
        "trace_id": row["trace_id"],
        "observation_id": row["observation_id"],
        "tool_name": row["name"].removeprefix("tool:"),
        "status": row["status"],
        "duration_ms": duration_ms(row),
    } for row in tools[:5]]
    return {"rows": rows, "requested_count": 5}


def query10(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for status in sorted({row["status"] for row in traces}):
        group = [row for row in traces if row["status"] == status]
        rows.append({
            "status": status,
            "trace_count": len(group),
            "mean_duration_ms": rounded(mean(row["duration_ms"] for row in group)),
            "mean_tool_count": rounded(mean(row["metadata"].get("tool_count", 0) for row in group)),
            "mean_input_tokens": rounded(mean((row.get("usage") or {}).get("input", 0) for row in group)),
            "mean_output_tokens": rounded(mean((row.get("usage") or {}).get("output", 0) for row in group)),
        })
    return {"rows": rows, "interpretation_limit": "Descriptive comparison only; this small non-random sample does not support a significance claim."}


def query11(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for failed in [row for row in traces if row["status"] == "error"]:
        failed_end = parse_time(failed["end_time"])
        later = [row for row in traces if row["session_id"] == failed["session_id"] and row["status"] == "success" and parse_time(row["start_time"]) > failed_end]
        later.sort(key=lambda row: (row["start_time"], row["trace_id"]))
        recovered = later[0] if later else None
        rows.append({
            "failed_trace_id": failed["trace_id"],
            "session_id": failed["session_id"],
            "later_success_found": recovered is not None,
            "next_success_trace_id": recovered["trace_id"] if recovered else None,
            "gap_seconds": rounded((parse_time(recovered["start_time"]) - failed_end).total_seconds()) if recovered else None,
        })
    return {"rows": rows, "failures_with_later_success": sum(row["later_success_found"] for row in rows), "warning": "A later success does not by itself prove an automatic retry or full task recovery."}


def query12(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[trace["session_id"]].append(trace)
    sessions = []
    for session_id, session_traces in sorted(grouped.items()):
        session_traces.sort(key=lambda row: (row["start_time"], row["trace_id"]))
        attempts = [{"sequence": i, "turn": row["metadata"].get("turn"), "status": row["status"], "start_time": row["start_time"], "duration_ms": row["duration_ms"], "trace_id": row["trace_id"]} for i, row in enumerate(session_traces, start=1)]
        sessions.append({"session_id": session_id, "status_sequence": [row["status"] for row in attempts], "attempts": attempts})
    return {"sessions": sessions, "session_count": len(sessions)}


def query13(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for trace in traces:
        output = trace.get("output")
        available = isinstance(output, str) and bool(output.strip())
        rows.append({"trace_id": trace["trace_id"], "output_available": available, "unicode_character_count": len(output) if available else None})
    return {"rows": rows, "available": sum(row["output_available"] for row in rows), "missing": sum(not row["output_available"] for row in rows)}


def query14(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    text_by_trace: dict[str, list[str]] = defaultdict(list)
    for trace in traces:
        for field in (trace.get("input"), trace.get("output")):
            if isinstance(field, str):
                text_by_trace[trace["trace_id"]].append(field)
    for row in observations:
        for field in (row.get("input"), row.get("output")):
            if isinstance(field, str):
                text_by_trace[row["trace_id"]].append(field)
    rows = []
    for trace in traces:
        versions = sorted({match.lower() for text in text_by_trace[trace["trace_id"]] for match in NOTEBOOK_PATTERN.findall(text)})
        rows.append({"trace_id": trace["trace_id"], "notebook_versions": versions, "version_count": len(versions)})
    all_versions = sorted({version for row in rows for version in row["notebook_versions"]})
    return {"rows": rows, "all_notebook_versions": all_versions}


def query15(traces: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    trace_ids = {row["trace_id"] for row in traces}
    observation_ids = {row["observation_id"] for row in observations}
    roots = Counter(row["trace_id"] for row in observations if row["parent_observation_id"] is None)
    broken_trace_links = [row["observation_id"] for row in observations if row["trace_id"] not in trace_ids]
    broken_parent_links = [row["observation_id"] for row in observations if row["parent_observation_id"] is not None and row["parent_observation_id"] not in observation_ids]
    bad_times = [row.get("trace_id") or row.get("observation_id") for row in traces + observations if parse_time(row["start_time"]) > parse_time(row["end_time"])]
    return {
        "trace_count": len(traces),
        "observation_count": len(observations),
        "observation_type_counts": dict(sorted(Counter(row["type"] for row in observations).items())),
        "traces_missing_model": sorted(row["trace_id"] for row in traces if not row.get("model") or row.get("model") == "unknown"),
        "traces_missing_output": sorted(row["trace_id"] for row in traces if not isinstance(row.get("output"), str) or not row["output"].strip()),
        "traces_missing_usage": sorted(row["trace_id"] for row in traces if not isinstance(row.get("usage"), dict)),
        "root_span_count_by_trace": dict(sorted(roots.items())),
        "traces_without_exactly_one_root": sorted(trace_id for trace_id in trace_ids if roots[trace_id] != 1),
        "broken_trace_links": sorted(broken_trace_links),
        "broken_parent_links": sorted(broken_parent_links),
        "invalid_timestamp_rows": sorted(bad_times),
    }


QUERY_FUNCTIONS: dict[str, Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]] = {
    f"query{i:02d}": globals()[f"query{i:02d}"] for i in range(1, 16)
}


def run_task(task_id: str, data_dir: Path) -> dict[str, Any]:
    if task_id not in QUERY_FUNCTIONS:
        raise ValueError(f"Unknown task: {task_id}")
    traces, observations = load_data(data_dir)
    return QUERY_FUNCTIONS[task_id](traces, observations)


def cli(default_task_id: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task_id",
        nargs="?" if default_task_id else None,
        default=default_task_id,
        choices=sorted(QUERY_FUNCTIONS),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("canonical_data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    answer = run_task(args.task_id, args.data_dir.resolve())
    payload = json.dumps(answer, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
