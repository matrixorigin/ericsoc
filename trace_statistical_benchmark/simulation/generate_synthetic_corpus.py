#!/usr/bin/env python3
"""Generate reproducible, relationally valid synthetic agent traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


GENERATOR_VERSION = "synthetic-trace-generator-v1"
SCHEMA_VERSION = "2.0.0"
MODEL_LIBRARY = [
    {"slug": "astra", "name": "Astra", "provider": "matrixorigin", "ability": 0.78, "latency": 1.10, "cost": 1.00, "tool_skill": 0.89},
    {"slug": "codex", "name": "Codex", "provider": "openai", "ability": 0.84, "latency": 0.92, "cost": 1.25, "tool_skill": 0.93},
]
TASK_FAMILIES = ["data_analysis", "code_debugging", "retrieval", "planning", "notebook_editing"]
TOOL_NAMES = ["read_file", "python", "search", "write_file", "shell"]
DIFFICULTY = {
    "easy": {"weight": 0.38, "penalty": -0.09, "tools": (0, 2), "generations": (1, 2)},
    "medium": {"weight": 0.34, "penalty": 0.04, "tools": (1, 4), "generations": (1, 3)},
    "hard": {"weight": 0.28, "penalty": 0.18, "tools": (2, 7), "generations": (2, 5)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=250)
    parser.add_argument("--model-count", type=int, choices=(2,), default=2)
    parser.add_argument("--target-trace-count", type=int)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--bootstrap-repetitions", type=int, default=300)
    parser.add_argument("--permutation-repetitions", type=int, default=300)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def bounded_probability(value: float) -> float:
    return max(0.01, min(0.99, value))


def weighted_choice(rng: random.Random, values: dict[str, dict[str, Any]]) -> str:
    point = rng.random() * sum(row["weight"] for row in values.values())
    running = 0.0
    for name, row in values.items():
        running += row["weight"]
        if point <= running:
            return name
    return next(reversed(values))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CorpusBuilder:
    def __init__(self, task_count: int, model_count: int, seed: int, target_trace_count: int | None = None) -> None:
        if task_count < 4:
            raise ValueError("task_count must be at least 4")
        self.task_count = task_count
        self.total_sessions = task_count * model_count
        self.completed_sessions = 0
        self.target_trace_count = target_trace_count
        if target_trace_count is not None and not self.total_sessions <= target_trace_count <= self.total_sessions * 4:
            raise ValueError("target_trace_count must allow between one and four attempts per session")
        self.seed = seed
        self.rng = random.Random(seed)
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.models = []
        for index, row in enumerate(MODEL_LIBRARY[:model_count], 1):
            self.models.append({
                "model_id": stable_id("model", seed, row["slug"]),
                "provider": row["provider"],
                "name": row["name"],
                "version": "sim-v1",
                "configuration": {
                    "temperature": 0.2,
                    "synthetic_parameters": {key: row[key] for key in ("ability", "latency", "cost", "tool_skill")},
                },
                **{f"_{key}": row[key] for key in ("slug", "ability", "latency", "cost", "tool_skill")},
            })
        self.tables: dict[str, list[dict[str, Any]]] = {
            name: [] for name in ("models", "tasks", "sessions", "traces", "observations", "generations", "tool_calls", "model_answers", "annotations")
        }
        self.truth: dict[str, Any] = {
            "generator_version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "requested_task_count": task_count,
            "requested_trace_count": target_trace_count,
            "model_count": model_count,
            "models": {},
            "drift": {"starts_at_task_fraction": 0.65, "latency_multiplier": 1.35, "success_probability_change": -0.08},
        }

    def build(self) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        self.tables["models"] = [{key: value for key, value in row.items() if not key.startswith("_")} for row in self.models]
        latest_trace: dict[tuple[str, str], dict[str, Any]] = {}
        for task_index in range(1, self.task_count + 1):
            task = self._make_task(task_index)
            self.tables["tasks"].append(task)
            for model_index, model in enumerate(self.models, 1):
                final = self._make_session_attempts(task, task_index, model, model_index)
                latest_trace[(task["task_id"], model["model_id"])] = final
        self._make_annotations(latest_trace)
        self._finish_truth()
        if self.target_trace_count is not None and len(self.tables["traces"]) != self.target_trace_count:
            raise RuntimeError(f"Generated {len(self.tables['traces'])} traces; expected {self.target_trace_count}")
        return self.tables, self.truth

    def _make_task(self, index: int) -> dict[str, Any]:
        difficulty = weighted_choice(self.rng, DIFFICULTY)
        family = TASK_FAMILIES[(index - 1) % len(TASK_FAMILIES)]
        notebook_version = f"hourly_h3_analysis_v{(index - 1) % 5:02d}.ipynb"
        task_id = stable_id("task", self.seed, index)
        return {
            "task_id": task_id,
            "task_family": family,
            "difficulty": difficulty,
            "prompt_version": "synthetic-prompt-v1",
            "prompt_text": f"Complete {family} task {index} using {notebook_version} and report the result.",
            "reference_outcome": {"expected_artifact": notebook_version, "task_number": index},
            "mapping_status": "synthetic-controlled-task",
            "metadata": {"task_number": index, "notebook_version": notebook_version, "synthetic": True},
        }

    def _make_session_attempts(self, task: dict[str, Any], task_index: int, model: dict[str, Any], model_index: int) -> dict[str, Any]:
        session_id = stable_id("session", self.seed, task_index, model["_slug"])
        base_time = self.start + timedelta(minutes=(task_index - 1) * 12 + model_index * 2)
        difficulty = task["difficulty"]
        drifted = task_index > math.floor(self.task_count * 0.65)
        failure_probability = 1 - model["_tool_skill"] + DIFFICULTY[difficulty]["penalty"]
        if drifted:
            failure_probability += 0.08
        first_success = self.rng.random() > bounded_probability(failure_probability)
        retry_count = 0 if first_success else self.rng.choices([0, 1, 2, 3], weights=[0.22, 0.38, 0.27, 0.13], k=1)[0]
        if self.target_trace_count is not None:
            generated_extras = len(self.tables["traces"]) - self.completed_sessions
            remaining_extras = self.target_trace_count - self.total_sessions - generated_extras
            remaining_sessions = self.total_sessions - self.completed_sessions - 1
            minimum_now = max(0, remaining_extras - remaining_sessions * 3)
            maximum_now = min(3, remaining_extras)
            retry_count = max(minimum_now, min(retry_count, maximum_now))
            if retry_count:
                first_success = False

        attempts: list[dict[str, Any]] = []
        cursor = base_time
        for attempt_number in range(1, retry_count + 2):
            if attempt_number == 1:
                success = first_success
            elif attempt_number <= retry_count:
                success = False
            else:
                recovery_probability = bounded_probability(0.48 + 0.13 * attempt_number + 0.08 * model["_tool_skill"])
                success = self.rng.random() < recovery_probability
            trace = self._make_trace(task, task_index, model, session_id, attempt_number, cursor, success, drifted)
            attempts.append(trace)
            cursor = datetime.fromisoformat(trace["end_time"].replace("Z", "+00:00")) + timedelta(seconds=self.rng.randint(20, 120))

        session_end = datetime.fromisoformat(attempts[-1]["end_time"].replace("Z", "+00:00"))
        self.tables["sessions"].append({
            "session_id": session_id,
            "source": "synthetic-controlled",
            "start_time": iso(base_time),
            "end_time": iso(session_end),
            "metadata": {"task_id": task["task_id"], "model_id": model["model_id"], "trace_count": len(attempts)},
        })
        self.completed_sessions += 1
        return attempts[-1]

    def _make_trace(
        self,
        task: dict[str, Any],
        task_index: int,
        model: dict[str, Any],
        session_id: str,
        attempt_number: int,
        start_time: datetime,
        success: bool,
        drifted: bool,
    ) -> dict[str, Any]:
        trace_id = stable_id("trace", self.seed, task_index, model["_slug"], attempt_number)
        difficulty = task["difficulty"]
        generation_count = self.rng.randint(*DIFFICULTY[difficulty]["generations"])
        tool_count = self.rng.randint(*DIFFICULTY[difficulty]["tools"])
        if attempt_number > 1:
            tool_count += 1
        # A successful agent can recover from an intermediate tool error.
        recovered_tool_error = success and tool_count > 0 and self.rng.random() < 0.08
        tool_failure_target = 1 if recovered_tool_error else (0 if success else max(1, math.ceil(tool_count * 0.35)))
        tool_failures = set(self.rng.sample(range(tool_count), min(tool_count, tool_failure_target))) if tool_count else set()
        latency_base = {"easy": 8_000, "medium": 18_000, "hard": 34_000}[difficulty]
        duration_ms = latency_base * model["_latency"] * self.rng.lognormvariate(0, 0.38)
        if drifted:
            duration_ms *= 1.35
        if not success:
            duration_ms *= 1.18
        duration_ms = max(900, round(duration_ms))
        end_time = start_time + timedelta(milliseconds=duration_ms)
        input_tokens = max(120, round((620 + tool_count * 220 + generation_count * 310) * self.rng.lognormvariate(0, 0.24)))
        output_tokens = max(20, round((110 + generation_count * 75) * self.rng.lognormvariate(0, 0.2)))
        cache_read = round(input_tokens * self.rng.choice([0, 0.1, 0.25, 0.5]))
        cost = round((input_tokens * 0.0000015 + output_tokens * 0.000006) * model["_cost"], 6)
        notebook = task["metadata"]["notebook_version"]
        output_available = success or self.rng.random() < 0.18
        output = f"Completed {notebook}; task {task_index} finished with attempt {attempt_number}." if output_available else None
        error = None if success else {"type": self.rng.choice(["tool_error", "timeout", "transport_error"]), "message": "Synthetic controlled failure", "retryable": True}

        trace = {
            "trace_id": trace_id,
            "session_id": session_id,
            "task_id": task["task_id"],
            "model_id": model["model_id"],
            "model": model["name"],
            "name": f"{model['_slug']}-task-{task_index:05d}-attempt-{attempt_number}",
            "attempt_number": attempt_number,
            "status": "success" if success else "error",
            "start_time": iso(start_time),
            "end_time": iso(end_time),
            "duration_ms": duration_ms,
            "usage": {"input": input_tokens, "output": output_tokens, "cache_read": cache_read},
            "cost": cost,
            "input": task["prompt_text"],
            "output": output,
            "error": error,
            "source": "synthetic-controlled",
            "metadata": {
                "turn": task_index,
                "tool_count": tool_count,
                "llm_rounds": generation_count,
                "difficulty": difficulty,
                "drifted_period": drifted,
                "synthetic": True,
            },
        }
        self.tables["traces"].append(trace)
        self._make_observations(trace, model, generation_count, tool_count, tool_failures)

        correctness_probability = model["_ability"] - DIFFICULTY[difficulty]["penalty"] - 0.06 * len(tool_failures)
        # Keep a detectable within-task relationship between longer agent paths and answer quality.
        correctness_probability -= 0.025 * max(0, tool_count + generation_count - 2)
        if not success:
            correctness_probability -= 0.28
        correct = self.rng.random() < bounded_probability(correctness_probability)
        self.tables["model_answers"].append({
            "answer_id": stable_id("answer", trace_id),
            "task_id": task["task_id"],
            "trace_id": trace_id,
            "model_id": model["model_id"],
            "answer_text": output,
            "parsed_answer": {"completed": bool(output_available), "notebook": notebook} if output_available else None,
            "correct": correct,
            "grading_version": "synthetic-known-truth-v1",
            "metadata": {"synthetic": True, "correctness_probability": round(bounded_probability(correctness_probability), 6)},
        })
        return trace

    def _make_observations(
        self,
        trace: dict[str, Any],
        model: dict[str, Any],
        generation_count: int,
        tool_count: int,
        tool_failures: set[int],
    ) -> None:
        start = datetime.fromisoformat(trace["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(trace["end_time"].replace("Z", "+00:00"))
        root_id = stable_id("obs-root", trace["trace_id"])
        root = {
            "observation_id": root_id,
            "trace_id": trace["trace_id"],
            "parent_observation_id": None,
            "type": "span",
            "name": "agent-run",
            "status": trace["status"],
            "start_time": trace["start_time"],
            "end_time": trace["end_time"],
            "input": trace["input"],
            "output": trace["output"],
            "metadata": {"synthetic": True},
        }
        self.tables["observations"].append(root)
        total_children = generation_count + tool_count
        step = max(1, int(trace["duration_ms"] / max(1, total_children + 1)))
        child_cursor = start + timedelta(milliseconds=50)
        generation_ids: list[str] = []
        for index in range(generation_count):
            duration = max(50, min(step, round(step * self.rng.uniform(0.55, 1.2))))
            child_end = min(end, child_cursor + timedelta(milliseconds=duration))
            observation_id = stable_id("obs-gen", trace["trace_id"], index + 1)
            parent = root_id if index == 0 else generation_ids[-1]
            generation_ids.append(observation_id)
            usage_input = max(20, round(trace["usage"]["input"] / generation_count * self.rng.uniform(0.75, 1.25)))
            usage_output = max(5, round(trace["usage"]["output"] / generation_count * self.rng.uniform(0.75, 1.25)))
            row = {
                "observation_id": observation_id,
                "trace_id": trace["trace_id"],
                "parent_observation_id": parent,
                "type": "generation",
                "name": f"llm-round-{index + 1}",
                "model": model["name"],
                "status": "success",
                "start_time": iso(child_cursor),
                "end_time": iso(child_end),
                "input": f"Reasoning step {index + 1} for {trace['input']}",
                "output": "Continue with tools" if index + 1 < generation_count else trace["output"],
                "usage": {"input": usage_input, "output": usage_output},
                "metadata": {"round": index + 1, "synthetic": True},
            }
            self.tables["observations"].append(row)
            self.tables["generations"].append({
                "observation_id": observation_id,
                "trace_id": trace["trace_id"],
                "model_id": model["model_id"],
                "input": row["input"],
                "output": row["output"],
                "usage": row["usage"],
                "duration_ms": round((child_end - child_cursor).total_seconds() * 1000),
                "metadata": row["metadata"],
            })
            child_cursor = min(end, child_end + timedelta(milliseconds=10))

        for index in range(tool_count):
            duration = max(30, min(step, round(step * self.rng.lognormvariate(-0.25, 0.55))))
            child_end = min(end, child_cursor + timedelta(milliseconds=duration))
            observation_id = stable_id("obs-tool", trace["trace_id"], index + 1)
            tool_name = TOOL_NAMES[(index + int(trace["metadata"]["turn"])) % len(TOOL_NAMES)]
            failed = index in tool_failures
            status = "error" if failed else "success"
            parent = generation_ids[min(index, len(generation_ids) - 1)] if generation_ids else root_id
            row = {
                "observation_id": observation_id,
                "trace_id": trace["trace_id"],
                "parent_observation_id": parent,
                "type": "tool",
                "name": f"tool:{tool_name}",
                "status": status,
                "start_time": iso(child_cursor),
                "end_time": iso(child_end),
                "input": {"task": trace["task_id"], "step": index + 1},
                "output": None if failed else f"Synthetic {tool_name} result",
                "metadata": {"tool_index": index + 1, "synthetic": True},
            }
            self.tables["observations"].append(row)
            self.tables["tool_calls"].append({
                "observation_id": observation_id,
                "trace_id": trace["trace_id"],
                "tool_name": tool_name,
                "tool_index": index + 1,
                "status": status,
                "input": row["input"],
                "output": row["output"],
                "duration_ms": round((child_end - child_cursor).total_seconds() * 1000),
                "error": {"type": "synthetic_tool_failure"} if failed else None,
                "metadata": row["metadata"],
            })
            child_cursor = min(end, child_end + timedelta(milliseconds=10))

    def _make_annotations(self, latest_trace: dict[tuple[str, str], dict[str, Any]]) -> None:
        selected_model = self.models[0]["model_id"]
        answer_by_trace = {row["trace_id"]: row for row in self.tables["model_answers"]}
        task_order = {row["task_id"]: row["metadata"]["task_number"] for row in self.tables["tasks"]}
        revision_count = 0
        for (task_id, model_id), trace in sorted(latest_trace.items(), key=lambda item: task_order[item[0][0]]):
            if model_id != selected_model:
                continue
            truth = bool(answer_by_trace[trace["trace_id"]]["correct"])
            base_time = datetime.fromisoformat(trace["end_time"].replace("Z", "+00:00")) + timedelta(hours=2)
            for reviewer_index, reviewer in enumerate(("synthetic-reviewer-a", "synthetic-reviewer-b")):
                accuracy = 0.88
                label = truth if self.rng.random() < accuracy else not truth
                labeled = base_time + timedelta(minutes=reviewer_index * 7 + self.rng.randint(1, 12))
                ingested = labeled + timedelta(seconds=self.rng.randint(1, 180))
                annotation_id = stable_id("ann", trace["trace_id"], reviewer, 1)
                annotation = {
                    "annotation_id": annotation_id,
                    "target_type": "trace",
                    "target_id": trace["trace_id"],
                    "trace_id": trace["trace_id"],
                    "label_schema_version": "1.0.0",
                    "label_json": {"answer_correct": label, "confidence": round(self.rng.uniform(0.65, 0.99), 2), "error_type": None, "severity": "none" if label else "medium"},
                    "annotator_id": reviewer,
                    "annotator_type": "rule",
                    "labeled_at": iso(labeled),
                    "ingested_at": iso(ingested),
                    "source": "synthetic-annotation-simulation",
                    "supersedes_annotation_id": None,
                    "retracted": False,
                    "comment": "Synthetic reviewer event; not a human judgment.",
                    "metadata": {"synthetic": True, "reviewer_accuracy_parameter": accuracy},
                }
                self.tables["annotations"].append(annotation)
                if self.rng.random() < 0.08:
                    revision_time = labeled + timedelta(minutes=self.rng.randint(10, 180))
                    revision = dict(annotation)
                    revision["annotation_id"] = stable_id("ann", trace["trace_id"], reviewer, 2)
                    revision["label_json"] = dict(annotation["label_json"], answer_correct=not label, confidence=round(self.rng.uniform(0.8, 1.0), 2))
                    revision["labeled_at"] = iso(revision_time)
                    revision["ingested_at"] = iso(revision_time + timedelta(seconds=self.rng.randint(1, 180)))
                    revision["supersedes_annotation_id"] = annotation_id
                    revision["comment"] = "Synthetic revision event; not a human judgment."
                    self.tables["annotations"].append(revision)
                    revision_count += 1

        # Small smoke profiles still need one revision to exercise history queries.
        if revision_count == 0 and self.tables["annotations"]:
            annotation = self.tables["annotations"][0]
            revision_time = datetime.fromisoformat(annotation["labeled_at"].replace("Z", "+00:00")) + timedelta(minutes=30)
            revision = dict(annotation)
            revision["annotation_id"] = stable_id("ann", annotation["trace_id"], annotation["annotator_id"], "fallback-revision")
            revision["label_json"] = dict(annotation["label_json"], answer_correct=not annotation["label_json"]["answer_correct"], confidence=0.9)
            revision["labeled_at"] = iso(revision_time)
            revision["ingested_at"] = iso(revision_time + timedelta(seconds=30))
            revision["supersedes_annotation_id"] = annotation["annotation_id"]
            revision["comment"] = "Synthetic revision event; not a human judgment."
            self.tables["annotations"].append(revision)

    def _finish_truth(self) -> None:
        traces = self.tables["traces"]
        answers = self.tables["model_answers"]
        ordered_traces = sorted(traces, key=lambda row: (row["start_time"], row["trace_id"]))
        first_drift_index = next((index for index, row in enumerate(ordered_traces, 1) if row["metadata"]["drifted_period"]), None)
        if first_drift_index is not None:
            first_drift = ordered_traces[first_drift_index - 1]
            self.truth["drift"].update({
                "first_drifted_trace_index_one_based": first_drift_index,
                "first_drifted_trace_id": first_drift["trace_id"],
                "first_drifted_trace_start_time": first_drift["start_time"],
            })
        for model in self.models:
            model_traces = [row for row in traces if row["model_id"] == model["model_id"]]
            latest_answer_by_task: dict[str, dict[str, Any]] = {}
            trace_attempt = {row["trace_id"]: row["attempt_number"] for row in model_traces}
            for row in answers:
                if row["model_id"] == model["model_id"]:
                    current = latest_answer_by_task.get(row["task_id"])
                    if current is None or trace_attempt[row["trace_id"]] > trace_attempt[current["trace_id"]]:
                        latest_answer_by_task[row["task_id"]] = row
            self.truth["models"][model["model_id"]] = {
                "name": model["name"],
                "trace_count": len(model_traces),
                "success_rate": round(sum(row["status"] == "success" for row in model_traces) / len(model_traces), 6),
                "final_answer_accuracy": round(sum(row["correct"] for row in latest_answer_by_task.values()) / len(latest_answer_by_task), 6),
                "median_cost_parameter": model["_cost"],
                "latency_multiplier_parameter": model["_latency"],
            }
        self.truth["counts"] = {name: len(rows) for name, rows in self.tables.items()}
        self.truth["trace_status"] = dict(sorted(Counter(row["status"] for row in traces).items()))
        self.truth["tool_status"] = dict(sorted(Counter(row["status"] for row in self.tables["tool_calls"]).items()))
        self.truth["difficulty"] = dict(sorted(Counter(row["difficulty"] for row in self.tables["tasks"]).items()))
        self.truth["notes"] = [
            "All records are synthetic and generated by deterministic Python.",
            "Astra means Matrix Origin's Astra agent system; Codex means OpenAI Codex.",
            "System names are simulation labels, not measurements of the real products.",
            "Synthetic annotations are generated reviewer behavior, not human labels.",
        ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.replace:
            raise SystemExit(f"Output directory is not empty: {output_dir}. Add --replace.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    tables, truth = CorpusBuilder(args.task_count, args.model_count, args.seed, args.target_trace_count).build()
    for name, rows in tables.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    config = {
        "selected_models": [row["model_id"] for row in tables["models"][:2]],
        "bootstrap_seed": args.seed,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "permutation_repetitions": args.permutation_repetitions,
        "minimum_change_point_window": max(5, min(50, args.task_count // 5)),
        "annotation_label_field": "answer_correct",
    }
    (output_dir / "benchmark_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "private_ground_truth.json").write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = {path.name: file_sha256(path) for path in sorted(output_dir.glob("*.json*"))}
    manifest = {
        "dataset_version": f"synthetic-agent-traces-v1-seed-{args.seed}",
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "counts": truth["counts"],
        "checksums_sha256": checksums,
        "contains_real_user_data": False,
        "contains_real_model_measurements": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
