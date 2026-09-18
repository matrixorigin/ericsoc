#!/usr/bin/env python3
"""Deterministic reference implementations for trace queries 16-30."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable


TABLES = (
    "models", "tasks", "sessions", "traces", "observations",
    "generations", "tool_calls", "model_answers", "annotations",
)
DEFAULT_CONFIG = {
    "bootstrap_seed": 20260914,
    "bootstrap_repetitions": 1000,
    "permutation_repetitions": 1000,
    "minimum_change_point_window": 5,
    "annotation_label_field": "answer_correct",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_bundle(data_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    bundle = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in TABLES}
    config = dict(DEFAULT_CONFIG)
    config_path = data_dir / "benchmark_config.json"
    if config_path.exists():
        config.update(json.loads(config_path.read_text(encoding="utf-8")))
    model_ids = sorted(row["model_id"] for row in bundle["models"] if row.get("model_id"))
    config.setdefault("selected_models", model_ids[:2])
    return bundle, config


def rounded(value: float | None, digits: int = 6) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, digits)


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_interval(values: list[float], statistic: Callable[[list[float]], float], seed: int, repetitions: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    samples = [statistic([rng.choice(values) for _ in values]) for _ in range(repetitions)]
    return [rounded(percentile(samples, 0.025)), rounded(percentile(samples, 0.975))]


def normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def two_sided_normal_p(z_value: float) -> float:
    # erfc keeps small tail probabilities distinguishable longer than 1 - CDF.
    return min(1.0, math.erfc(abs(z_value) / math.sqrt(2)))


def selected_models(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> tuple[str | None, str | None]:
    values = list(config.get("selected_models") or [])
    if len(values) < 2:
        values = sorted(row["model_id"] for row in bundle["models"] if row.get("model_id"))[:2]
    return (values[0], values[1]) if len(values) >= 2 else (None, None)


def canonical_answers(bundle: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], dict[str, Any]]:
    traces = {row["trace_id"]: row for row in bundle["traces"]}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for answer in bundle["model_answers"]:
        if answer.get("task_id") and answer.get("model_id"):
            grouped[(answer["task_id"], answer["model_id"])].append(answer)
    chosen = {}
    for key, rows in grouped.items():
        chosen[key] = max(
            rows,
            key=lambda row: (
                int((traces.get(row.get("trace_id")) or {}).get("attempt_number") or 0),
                str((traces.get(row.get("trace_id")) or {}).get("end_time") or ""),
                row["answer_id"],
            ),
        )
    return chosen


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right))
    return None if denominator == 0 else numerator / denominator


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(average_ranks(left), average_ranks(right))


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2
    if total == 0:
        return 1.0
    lower = max(0, col1 - row2)
    upper = min(row1, col1)

    def probability(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(total, col1)

    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(lower, upper + 1) if probability(x) <= observed + 1e-15))


def query16(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    tables = {name: {row.get(f"{name[:-1]}_id"): row for row in bundle[name]} for name in ("models", "tasks", "sessions")}
    traces = {row["trace_id"]: row for row in bundle["traces"]}
    roots = defaultdict(list)
    for row in bundle["observations"]:
        if row.get("parent_observation_id") is None:
            roots[row.get("trace_id")].append(row["observation_id"])
    seen = Counter((row.get("task_id"), row.get("model_id"), row.get("trace_id")) for row in bundle["model_answers"])
    rows, issues = [], Counter()
    for answer in sorted(bundle["model_answers"], key=lambda row: row["answer_id"]):
        trace = traces.get(answer.get("trace_id"))
        session = tables["sessions"].get(trace.get("session_id")) if trace else None
        row_issues = []
        for label, exists in (
            ("missing_trace", trace is not None),
            ("missing_task", answer.get("task_id") in tables["tasks"]),
            ("missing_model", answer.get("model_id") in tables["models"]),
            ("missing_session", session is not None),
        ):
            if not exists:
                row_issues.append(label)
        root_ids = roots.get(answer.get("trace_id"), [])
        if len(root_ids) != 1:
            row_issues.append("ambiguous_root_observation")
        if seen[(answer.get("task_id"), answer.get("model_id"), answer.get("trace_id"))] > 1:
            row_issues.append("duplicate_answer_link")
        issues.update(row_issues)
        rows.append({
            "answer_id": answer["answer_id"], "task_id": answer.get("task_id"),
            "model_id": answer.get("model_id"), "session_id": trace.get("session_id") if trace else None,
            "trace_id": answer.get("trace_id"), "root_observation_ids": sorted(root_ids),
            "grading_version": answer.get("grading_version"), "issues": sorted(set(row_issues)),
        })
    return {"rows": rows, "answer_count": len(rows), "issue_counts": dict(sorted(issues.items()))}


def paired_binary_rows(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    model_a, model_b = selected_models(bundle, config)
    if not model_a or not model_b:
        return model_a, model_b, []
    answers = canonical_answers(bundle)
    tasks = sorted({task_id for task_id, model_id in answers if model_id == model_a} & {task_id for task_id, model_id in answers if model_id == model_b})
    rows = []
    for task_id in tasks:
        left, right = answers[(task_id, model_a)], answers[(task_id, model_b)]
        if isinstance(left.get("correct"), bool) and isinstance(right.get("correct"), bool):
            rows.append({"task_id": task_id, "model_a_correct": left["correct"], "model_b_correct": right["correct"], "answer_a": left, "answer_b": right})
    return model_a, model_b, rows


def query17(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    model_a, model_b, pairs = paired_binary_rows(bundle, config)
    if not pairs:
        return {"model_a": model_a, "model_b": model_b, "matched_tasks": 0, "warning": "At least two models with graded answers on shared tasks are required."}
    a_values = [int(row["model_a_correct"]) for row in pairs]
    b_values = [int(row["model_b_correct"]) for row in pairs]
    differences = [a - b for a, b in zip(a_values, b_values)]
    a_only = sum(a == 1 and b == 0 for a, b in zip(a_values, b_values))
    b_only = sum(a == 0 and b == 1 for a, b in zip(a_values, b_values))
    return {
        "model_a": model_a, "model_b": model_b, "matched_tasks": len(pairs),
        "accuracy_a": rounded(mean(a_values)), "accuracy_b": rounded(mean(b_values)),
        "paired_accuracy_difference_a_minus_b": rounded(mean(differences)),
        "difference_bootstrap_95_ci": bootstrap_interval(differences, mean, config["bootstrap_seed"], config["bootstrap_repetitions"]),
        "discordant_pairs": {"a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only},
        "exact_mcnemar_p_value": rounded(exact_mcnemar(a_only, b_only)),
        "task_ids": [row["task_id"] for row in pairs],
    }


def query18(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    model_a, model_b = selected_models(bundle, config)
    if not model_a or not model_b:
        return {"model_a": model_a, "model_b": model_b, "matched_successful_tasks": 0, "warning": "Two models are required."}
    answers, traces = canonical_answers(bundle), {row["trace_id"]: row for row in bundle["traces"]}
    task_ids = sorted({task for task, model in answers if model == model_a} & {task for task, model in answers if model == model_b})
    rows = []
    for task_id in task_ids:
        trace_a = traces.get(answers[(task_id, model_a)].get("trace_id"))
        trace_b = traces.get(answers[(task_id, model_b)].get("trace_id"))
        if trace_a and trace_b and trace_a.get("status") == trace_b.get("status") == "success":
            rows.append({"task_id": task_id, "latency_a_ms": trace_a["duration_ms"], "latency_b_ms": trace_b["duration_ms"], "difference_a_minus_b_ms": trace_a["duration_ms"] - trace_b["duration_ms"]})
    differences = [row["difference_a_minus_b_ms"] for row in rows]
    return {
        "model_a": model_a, "model_b": model_b, "matched_successful_tasks": len(rows),
        "median_difference_a_minus_b_ms": rounded(median(differences)) if differences else None,
        "bootstrap_95_ci": bootstrap_interval(differences, median, config["bootstrap_seed"], config["bootstrap_repetitions"]),
        "rows": rows,
        "warning": None if rows else "No matched successful tasks were available.",
    }


def query19(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    answers, traces = canonical_answers(bundle), {row["trace_id"]: row for row in bundle["traces"]}
    models = sorted({model_id for _, model_id in answers})
    common_tasks = set.intersection(*[{task for task, model in answers if model == model_id} for model_id in models]) if models else set()
    metrics = []
    for model_id in models:
        rows = [(answers[(task_id, model_id)], traces.get(answers[(task_id, model_id)].get("trace_id"))) for task_id in sorted(common_tasks)]
        graded = [int(answer["correct"]) for answer, _ in rows if isinstance(answer.get("correct"), bool)]
        complete = [(answer, trace) for answer, trace in rows if trace and trace.get("cost") is not None and trace.get("duration_ms") is not None]
        metrics.append({
            "model_id": model_id, "matched_tasks": len(common_tasks),
            "accuracy": rounded(mean(graded)) if graded else None,
            "median_cost": rounded(median(trace["cost"] for _, trace in complete)) if complete else None,
            "median_latency_ms": rounded(median(trace["duration_ms"] for _, trace in complete)) if complete else None,
            "complete_for_pareto": len(graded) == len(common_tasks) == len(complete) and bool(common_tasks),
        })
    for row in metrics:
        row["pareto_efficient"] = row["complete_for_pareto"] and not any(
            other["complete_for_pareto"]
            and other["accuracy"] >= row["accuracy"]
            and other["median_cost"] <= row["median_cost"]
            and other["median_latency_ms"] <= row["median_latency_ms"]
            and (other["accuracy"] > row["accuracy"] or other["median_cost"] < row["median_cost"] or other["median_latency_ms"] < row["median_latency_ms"])
            for other in metrics if other is not row
        )
    return {"common_task_count": len(common_tasks), "models": metrics, "frontier": [row["model_id"] for row in metrics if row["pareto_efficient"]], "warning": None if common_tasks else "No tasks were attempted by every model."}


def query20(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    model_a, model_b, pairs = paired_binary_rows(bundle, config)
    tasks = {row["task_id"]: row for row in bundle["tasks"]}
    strata: dict[str, list[float]] = {"easy": [], "hard": []}
    for row in pairs:
        difficulty = str((tasks.get(row["task_id"]) or {}).get("difficulty") or "").lower()
        if difficulty in strata:
            strata[difficulty].append(int(row["model_a_correct"]) - int(row["model_b_correct"]))
    result = {name: {"n": len(values), "accuracy_gap_a_minus_b": rounded(mean(values)) if values else None} for name, values in strata.items()}
    if not strata["easy"] or not strata["hard"]:
        return {"model_a": model_a, "model_b": model_b, "strata": result, "interaction_hard_minus_easy": None, "warning": "Matched graded tasks are required in both easy and hard strata."}
    interaction = mean(strata["hard"]) - mean(strata["easy"])
    rng = random.Random(config["bootstrap_seed"])
    boot = []
    for _ in range(config["bootstrap_repetitions"]):
        easy = [rng.choice(strata["easy"]) for _ in strata["easy"]]
        hard = [rng.choice(strata["hard"]) for _ in strata["hard"]]
        boot.append(mean(hard) - mean(easy))
    return {"model_a": model_a, "model_b": model_b, "strata": result, "interaction_hard_minus_easy": rounded(interaction), "bootstrap_95_ci": [rounded(percentile(boot, 0.025)), rounded(percentile(boot, 0.975))]}


def observation_depth(observation_id: str, observations: dict[str, dict[str, Any]]) -> int:
    depth, seen, current = 1, set(), observations.get(observation_id)
    while current and current.get("parent_observation_id") is not None:
        parent_id = current["parent_observation_id"]
        if parent_id in seen:
            return depth
        seen.add(parent_id)
        depth += 1
        current = observations.get(parent_id)
    return depth


def query21(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    answers, traces = canonical_answers(bundle), {row["trace_id"]: row for row in bundle["traces"]}
    observations = {row["observation_id"]: row for row in bundle["observations"]}
    by_trace = defaultdict(list)
    for row in bundle["observations"]:
        by_trace[row["trace_id"]].append(row)
    rows = []
    for answer in answers.values():
        if not isinstance(answer.get("correct"), bool) or answer.get("trace_id") not in traces:
            continue
        trace_rows = by_trace[answer["trace_id"]]
        max_depth = max((observation_depth(row["observation_id"], observations) for row in trace_rows), default=0)
        tool_count = sum(row.get("type") == "tool" for row in trace_rows)
        generation_count = sum(row.get("type") == "generation" for row in trace_rows)
        complexity = max_depth + tool_count + generation_count
        rows.append({"task_id": answer["task_id"], "trace_id": answer["trace_id"], "complexity_index": complexity, "correct": int(answer["correct"])})
    observed = spearman([row["complexity_index"] for row in rows], [row["correct"] for row in rows])
    if observed is None:
        return {"n": len(rows), "spearman_rho": None, "permutation_p_value": None, "rows": rows, "warning": "Variation in both complexity and correctness is required."}
    rng = random.Random(config["bootstrap_seed"])
    task_positions = defaultdict(list)
    for index, row in enumerate(rows):
        task_positions[row["task_id"]].append(index)
    extreme = 0
    for _ in range(config["permutation_repetitions"]):
        shuffled = [row["correct"] for row in rows]
        for positions in task_positions.values():
            task_values = [shuffled[index] for index in positions]
            rng.shuffle(task_values)
            for index, value in zip(positions, task_values):
                shuffled[index] = value
        permuted = spearman([row["complexity_index"] for row in rows], shuffled)
        extreme += permuted is not None and abs(permuted) >= abs(observed) - 1e-15
    return {"n": len(rows), "spearman_rho": rounded(observed), "permutation_p_value": rounded((extreme + 1) / (config["permutation_repetitions"] + 1)), "rows": rows, "complexity_definition": "maximum observation depth + tool count + generation count"}


def query22(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    failed_tools = {row["trace_id"] for row in bundle["tool_calls"] if row.get("status") != "success"}
    a = sum(row["trace_id"] in failed_tools and row.get("status") != "success" for row in bundle["traces"])
    b = sum(row["trace_id"] in failed_tools and row.get("status") == "success" for row in bundle["traces"])
    c = sum(row["trace_id"] not in failed_tools and row.get("status") != "success" for row in bundle["traces"])
    d = sum(row["trace_id"] not in failed_tools and row.get("status") == "success" for row in bundle["traces"])
    corrected = any(value == 0 for value in (a, b, c, d))
    aa, bb, cc, dd = (a + 0.5, b + 0.5, c + 0.5, d + 0.5) if corrected else (a, b, c, d)
    risk_exposed = aa / (aa + bb)
    risk_unexposed = cc / (cc + dd)
    rr = risk_exposed / risk_unexposed
    standard_error = math.sqrt(1 / aa - 1 / (aa + bb) + 1 / cc - 1 / (cc + dd))
    ci = [math.exp(math.log(rr) - 1.96 * standard_error), math.exp(math.log(rr) + 1.96 * standard_error)]
    return {"table": {"tool_failure_and_task_failure": a, "tool_failure_and_task_success": b, "no_tool_failure_and_task_failure": c, "no_tool_failure_and_task_success": d}, "risk_ratio": rounded(rr), "risk_ratio_95_ci": [rounded(ci[0]), rounded(ci[1])], "fisher_exact_p_value": rounded(fisher_exact_two_sided(a, b, c, d)), "zero_cell_correction": "Haldane-Anscombe 0.5" if corrected else None}


def query23(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    by_session = defaultdict(list)
    for trace in bundle["traces"]:
        by_session[trace["session_id"]].append(trace)
    units = []
    for session_id, traces in sorted(by_session.items()):
        traces.sort(key=lambda row: (row.get("attempt_number", 0), row.get("start_time", ""), row["trace_id"]))
        first_failure_index = next((i for i, row in enumerate(traces) if row.get("status") != "success"), None)
        if first_failure_index is None:
            continue
        later = traces[first_failure_index + 1:]
        units.append({"session_id": session_id, "original_failure_trace_id": traces[first_failure_index]["trace_id"], "retry_count": len(later), "recovered": any(row.get("status") == "success" for row in later)})
    grouped = []
    for retry_count in sorted({row["retry_count"] for row in units}):
        group = [row for row in units if row["retry_count"] == retry_count]
        grouped.append({"retry_count": retry_count, "original_failures": len(group), "recovered": sum(row["recovered"] for row in group), "recovery_rate": rounded(mean(int(row["recovered"]) for row in group))})
    if len(units) < 3 or len(grouped) < 2:
        return {"units": units, "strata": grouped, "cochran_armitage_z": None, "trend_p_value": None, "warning": "At least three independent original failures across two retry levels are required."}
    total = len(units)
    total_recovered = sum(row["recovered"] for row in units)
    pooled_rate = total_recovered / total
    weighted_score = sum(row["retry_count"] for row in units) / total
    numerator = sum(
        group["retry_count"] * (group["recovered"] - group["original_failures"] * pooled_rate)
        for group in grouped
    )
    denominator = math.sqrt(
        pooled_rate * (1 - pooled_rate)
        * sum(group["original_failures"] * (group["retry_count"] - weighted_score) ** 2 for group in grouped)
    )
    if denominator == 0:
        return {"units": units, "strata": grouped, "cochran_armitage_z": None, "trend_p_value": None, "warning": "The recovery outcome has no usable variation."}
    z_value = numerator / denominator
    return {"units": units, "strata": grouped, "cochran_armitage_z": rounded(z_value), "trend_p_value": rounded(two_sided_normal_p(z_value)), "warning": "Retry count is observational; the result is not causal."}


def gini(values: list[float]) -> float | None:
    values = sorted(value for value in values if value >= 0)
    if not values or sum(values) == 0:
        return None
    n = len(values)
    return (2 * sum((index + 1) * value for index, value in enumerate(values)) / (n * sum(values))) - (n + 1) / n


def query24(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    failed_names = [row["tool_name"] for row in bundle["tool_calls"] if row.get("status") != "success"]
    counts = Counter(failed_names)
    total = len(failed_names)
    if not total:
        return {"failed_tool_calls": 0, "tools": [], "hhi": None, "gini": None, "warning": "No failed tool calls were recorded."}

    def measures(sample: list[str]) -> tuple[float, float]:
        sample_counts = Counter(sample)
        denominator = len(sample)
        hhi_value = sum((count / denominator) ** 2 for count in sample_counts.values())
        return hhi_value, gini(list(sample_counts.values())) or 0.0

    hhi_value, gini_value = measures(failed_names)
    rng = random.Random(config["bootstrap_seed"])
    boot = [measures([rng.choice(failed_names) for _ in failed_names]) for _ in range(config["bootstrap_repetitions"])]
    return {"failed_tool_calls": total, "tools": [{"tool_name": name, "failures": count, "share": rounded(count / total)} for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))], "hhi": rounded(hhi_value), "hhi_bootstrap_95_ci": [rounded(percentile([row[0] for row in boot], 0.025)), rounded(percentile([row[0] for row in boot], 0.975))], "gini": rounded(gini_value), "gini_bootstrap_95_ci": [rounded(percentile([row[1] for row in boot], 0.025)), rounded(percentile([row[1] for row in boot], 0.975))]}


def query25(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    traces = sorted(bundle["traces"], key=lambda row: (row["start_time"], row["trace_id"]))
    minimum = int(config["minimum_change_point_window"])
    if len(traces) < minimum * 2:
        return {"trace_count": len(traces), "eligible_split_count": 0, "selected_change_point": None, "warning": f"At least {minimum * 2} ordered traces are required."}
    success_prefix = [0]
    latency_prefix = [0.0]
    latency_square_prefix = [0.0]
    for trace in traces:
        latency = float(trace["duration_ms"])
        success_prefix.append(success_prefix[-1] + int(trace.get("status") == "success"))
        latency_prefix.append(latency_prefix[-1] + latency)
        latency_square_prefix.append(latency_square_prefix[-1] + latency * latency)

    def summary(prefix: list[float], square_prefix: list[float], start: int, end: int) -> tuple[float, float]:
        count = end - start
        total = prefix[end] - prefix[start]
        average = total / count
        square_total = square_prefix[end] - square_prefix[start]
        squared_deviation = max(0.0, square_total - total * total / count)
        variance = squared_deviation / max(1, count - 1)
        return average, variance

    candidates = []
    for split in range(minimum, len(traces) - minimum + 1):
        before_count, after_count = split, len(traces) - split
        success_before = success_prefix[split] / before_count
        success_after = (success_prefix[-1] - success_prefix[split]) / after_count
        pooled = success_prefix[-1] / len(traces)
        success_se = math.sqrt(pooled * (1 - pooled) * (1 / before_count + 1 / after_count))
        success_p = two_sided_normal_p((success_after - success_before) / success_se) if success_se else 1.0
        latency_before, var_before = summary(latency_prefix, latency_square_prefix, 0, split)
        latency_after, var_after = summary(latency_prefix, latency_square_prefix, split, len(traces))
        latency_se = math.sqrt(var_before / before_count + var_after / after_count)
        latency_p = two_sided_normal_p((latency_after - latency_before) / latency_se) if latency_se else 1.0
        for metric, p_value, effect in (
            ("success_rate", success_p, success_after - success_before),
            ("mean_latency_ms", latency_p, latency_after - latency_before),
        ):
            candidates.append({"split_index": split, "split_time": traces[split]["start_time"], "metric": metric, "effect_after_minus_before": rounded(effect), "raw_p_value": p_value})
    tests = len(candidates)
    best_index = min(
        range(len(candidates)),
        key=lambda index: (
            min(1.0, candidates[index]["raw_p_value"] * tests),
            candidates[index]["split_index"],
            candidates[index]["metric"],
        ),
    )
    for row in candidates:
        row["bonferroni_p_value"] = rounded(min(1.0, row["raw_p_value"] * tests))
        row["raw_p_value"] = rounded(row["raw_p_value"])
    best = candidates[best_index]
    return {"trace_count": len(traces), "eligible_split_count": len(candidates) // 2, "tested_hypotheses": tests, "selected_change_point": best, "evidence_after_correction": best["bonferroni_p_value"] < 0.05, "warning": "This is a screening test; the selected point should be confirmed on new data."}


def annotation_issues(bundle: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    from datetime import datetime

    trace_ids = {row["trace_id"] for row in bundle["traces"]}
    observation_map = {row["observation_id"]: row for row in bundle["observations"]}
    session_ids = {row["session_id"] for row in bundle["sessions"]}
    annotation_map = {row["annotation_id"]: row for row in bundle["annotations"] if row.get("annotation_id")}
    duplicate_ids = {value for value, count in Counter(row.get("annotation_id") for row in bundle["annotations"]).items() if value and count > 1}
    issue_map: dict[str, set[str]] = defaultdict(set)
    required = {
        "annotation_id", "target_type", "target_id", "label_schema_version", "label_json",
        "annotator_id", "annotator_type", "labeled_at", "ingested_at", "source",
        "supersedes_annotation_id", "retracted",
    }
    parsed_times = {}
    for index, row in enumerate(bundle["annotations"], 1):
        annotation_id = row.get("annotation_id") or f"missing-id-row-{index}"
        reasons = issue_map[annotation_id]
        if annotation_id in duplicate_ids:
            reasons.add("duplicate_annotation_id")
        if missing := required - row.keys():
            reasons.update(f"missing_field:{field}" for field in missing)
        target_type, target_id = row.get("target_type"), row.get("target_id")
        if target_type == "trace" and target_id not in trace_ids:
            reasons.add("missing_trace_target")
        elif target_type == "observation" and target_id not in observation_map:
            reasons.add("missing_observation_target")
        elif target_type == "session" and target_id not in session_ids:
            reasons.add("missing_session_target")
        elif target_type not in {"trace", "observation", "session"}:
            reasons.add("invalid_target_type")
        if target_type == "trace" and row.get("trace_id") != target_id:
            reasons.add("trace_target_mismatch")
        if target_type == "observation" and target_id in observation_map and row.get("trace_id") != observation_map[target_id].get("trace_id"):
            reasons.add("trace_observation_mismatch")
        if target_type == "session" and row.get("trace_id") is not None:
            reasons.add("session_target_has_trace_id")
        if not row.get("annotator_id"):
            reasons.add("missing_annotator_id")
        if row.get("annotator_type") not in {"human", "llm", "rule", "imported"}:
            reasons.add("invalid_annotator_type")
        if not row.get("label_schema_version"):
            reasons.add("missing_label_schema_version")
        if not row.get("source"):
            reasons.add("missing_source")
        if not isinstance(row.get("retracted"), bool):
            reasons.add("invalid_retracted_flag")
        if not isinstance(row.get("label_json"), dict):
            reasons.add("invalid_label_json")
        else:
            confidence = row["label_json"].get("confidence")
            if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
                reasons.add("invalid_confidence")
        try:
            labeled = datetime.fromisoformat(row["labeled_at"].replace("Z", "+00:00"))
            ingested = datetime.fromisoformat(row["ingested_at"].replace("Z", "+00:00"))
            parsed_times[annotation_id] = labeled
            if ingested < labeled:
                reasons.add("ingested_before_labeled")
        except (KeyError, TypeError, ValueError):
            reasons.add("invalid_timestamp")
        predecessor = row.get("supersedes_annotation_id")
        if predecessor is not None and predecessor not in annotation_map:
            reasons.add("missing_predecessor")
        elif predecessor is not None:
            previous = annotation_map[predecessor]
            for field in ("target_type", "target_id", "annotator_id", "label_schema_version"):
                if previous.get(field) != row.get(field):
                    reasons.add(f"revision_changes:{field}")
            if predecessor in parsed_times and annotation_id in parsed_times and parsed_times[predecessor] >= parsed_times[annotation_id]:
                reasons.add("revision_not_later")

    for annotation_id in annotation_map:
        seen = []
        current = annotation_id
        while current in annotation_map:
            if current in seen:
                for member in seen[seen.index(current):]:
                    issue_map[member].add("revision_cycle")
                break
            seen.append(current)
            current = annotation_map[current].get("supersedes_annotation_id")
            if current is None:
                break
    return [
        {"annotation_id": annotation_id, "reasons": sorted(reasons)}
        for annotation_id, reasons in sorted(issue_map.items()) if reasons
    ]


def query26(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    issues = annotation_issues(bundle)
    counts = Counter(reason for row in issues for reason in row["reasons"])
    return {"annotation_count": len(bundle["annotations"]), "invalid_annotation_count": len(issues), "reason_counts": dict(sorted(counts.items())), "rows": issues}


def effective_annotations(bundle: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    invalid = annotation_issues(bundle)
    valid_ids = {row["annotation_id"] for row in bundle["annotations"]} - {row["annotation_id"] for row in invalid}
    rows = {row["annotation_id"]: row for row in bundle["annotations"] if row["annotation_id"] in valid_ids}
    children = defaultdict(list)
    for row in rows.values():
        if row.get("supersedes_annotation_id") in rows:
            children[row["supersedes_annotation_id"]].append(row["annotation_id"])
    grouped = defaultdict(list)
    for row in rows.values():
        grouped[(row["target_type"], row["target_id"], row["annotator_id"], row["label_schema_version"])].append(row)
    effective = []
    anomalies = [row for row in invalid if any("revision" in reason for reason in row["reasons"])]
    for key, group in sorted(grouped.items()):
        group_ids = {row["annotation_id"] for row in group}
        branches = sorted(parent for parent, child_ids in children.items() if parent in group_ids and len([child for child in child_ids if child in group_ids]) > 1)
        heads = [row for row in group if not any(row["annotation_id"] == other.get("supersedes_annotation_id") for other in group)]
        if branches or len(heads) != 1:
            anomalies.append({"target_type": key[0], "target_id": key[1], "annotator_id": key[2], "branch_points": branches, "head_ids": sorted(row["annotation_id"] for row in heads)})
            continue
        head = heads[0]
        if not head.get("retracted"):
            effective.append(head)
    return sorted(effective, key=lambda row: (row["target_type"], row["target_id"], row["annotator_id"])), anomalies


def query27(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    effective, anomalies = effective_annotations(bundle)
    return {"effective_label_count": len(effective), "rows": [{"annotation_id": row["annotation_id"], "target_type": row["target_type"], "target_id": row["target_id"], "annotator_id": row["annotator_id"], "label_json": row["label_json"]} for row in effective], "history_anomalies": anomalies}


def cohen_kappa(left: list[Any], right: list[Any]) -> tuple[float | None, float | None]:
    if not left or len(left) != len(right):
        return None, None
    agreement = mean(a == b for a, b in zip(left, right))
    labels = set(left) | set(right)
    expected = sum((left.count(label) / len(left)) * (right.count(label) / len(right)) for label in labels)
    kappa = None if expected == 1 else (agreement - expected) / (1 - expected)
    return agreement, kappa


def annotation_target_category(row: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]) -> str:
    traces = {trace["trace_id"]: trace for trace in bundle["traces"]}
    observations = {obs["observation_id"]: obs for obs in bundle["observations"]}
    tasks = {task["task_id"]: task for task in bundle["tasks"]}
    trace_id = row.get("trace_id")
    if row.get("target_type") == "observation" and row.get("target_id") in observations:
        trace_id = observations[row["target_id"]].get("trace_id")
    task_id = (traces.get(trace_id) or {}).get("task_id")
    task = tasks.get(task_id) or {}
    return str(task.get("task_family") or task.get("difficulty") or "unknown")


def agreement_rows(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> list[dict[str, Any]]:
    field = config["annotation_label_field"]
    effective, _ = effective_annotations(bundle)
    tasks = {row["task_id"]: row for row in bundle["tasks"]}
    trace_categories = {
        row["trace_id"]: str((tasks.get(row.get("task_id")) or {}).get("task_family") or (tasks.get(row.get("task_id")) or {}).get("difficulty") or "unknown")
        for row in bundle["traces"]
    }
    observation_traces = {row["observation_id"]: row.get("trace_id") for row in bundle["observations"]}
    by_target = defaultdict(dict)
    for row in effective:
        value = row.get("label_json", {}).get(field)
        if value is not None:
            trace_id = row.get("trace_id")
            if row.get("target_type") == "observation":
                trace_id = observation_traces.get(row.get("target_id"))
            category = trace_categories.get(trace_id, "unknown")
            by_target[(row["target_type"], row["target_id"])][row["annotator_id"]] = (value, category)
    pair_values = defaultdict(list)
    for target, labels in by_target.items():
        for left, right in itertools.combinations(sorted(labels), 2):
            pair_values[(left, right)].append({"target": f"{target[0]}:{target[1]}", "left": labels[left][0], "right": labels[right][0], "category": labels[left][1]})
    return [{"annotator_a": pair[0], "annotator_b": pair[1], "items": items} for pair, items in sorted(pair_values.items())]


def query28(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    output = []
    for pair in agreement_rows(bundle, config):
        items = pair.pop("items")
        agreement, kappa = cohen_kappa([row["left"] for row in items], [row["right"] for row in items])
        strata = []
        for category in sorted({row["category"] for row in items}):
            group = [row for row in items if row["category"] == category]
            raw, stratified_kappa = cohen_kappa([row["left"] for row in group], [row["right"] for row in group])
            strata.append({"category": category, "n": len(group), "raw_agreement": rounded(raw), "cohen_kappa": rounded(stratified_kappa)})
        output.append({**pair, "shared_targets": len(items), "raw_agreement": rounded(agreement), "cohen_kappa": rounded(kappa), "strata": strata})
    return {"label_field": config["annotation_label_field"], "annotator_pairs": output, "warning": None if output else "No annotator pair has comparable effective labels."}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def query29(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    rows, raw_p = [], []
    for pair in agreement_rows(bundle, config):
        items = pair.pop("items")
        left = [bool(row["left"]) for row in items]
        right = [bool(row["right"]) for row in items]
        left_only = sum(a and not b for a, b in zip(left, right))
        right_only = sum(not a and b for a, b in zip(left, right))
        p_value = exact_mcnemar(left_only, right_only)
        raw_p.append(p_value)
        rows.append({**pair, "shared_targets": len(items), "positive_rate_a": rounded(mean(left)), "positive_rate_b": rounded(mean(right)), "paired_difference_a_minus_b": rounded(mean(int(a) - int(b) for a, b in zip(left, right))), "discordant_a_positive_only": left_only, "discordant_b_positive_only": right_only, "raw_mcnemar_p_value": rounded(p_value)})
    for row, adjusted in zip(rows, holm_adjust(raw_p)):
        row["holm_adjusted_p_value"] = rounded(adjusted)
    return {"label_field": config["annotation_label_field"], "pairs": rows, "warning": None if rows else "No annotator pair has shared effective labels."}


def query30(bundle: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    from datetime import datetime

    annotations = {row["annotation_id"]: row for row in bundle["annotations"]}
    field = config["annotation_label_field"]

    def seconds(start: str, end: str) -> float:
        return (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()

    ingestion_lags, revision_lags, transitions = [], [], Counter()
    valid_events = 0
    for row in annotations.values():
        try:
            ingestion_lags.append(seconds(row["labeled_at"], row["ingested_at"]))
            valid_events += 1
        except (KeyError, TypeError, ValueError):
            continue
        predecessor = annotations.get(row.get("supersedes_annotation_id"))
        if predecessor:
            try:
                revision_lags.append(seconds(predecessor["labeled_at"], row["labeled_at"]))
            except (KeyError, TypeError, ValueError):
                pass
            old = predecessor.get("label_json", {}).get(field)
            new = "RETRACTED" if row.get("retracted") else row.get("label_json", {}).get(field)
            transitions[f"{old}->{new}"] += 1

    def timing(values: list[float]) -> dict[str, Any]:
        return {"count": len(values), "median_seconds": rounded(median(values)) if values else None, "p95_seconds": rounded(percentile(values, 0.95))}

    revision_count = sum(row.get("supersedes_annotation_id") is not None for row in annotations.values())
    return {"annotation_events": len(annotations), "valid_timing_events": valid_events, "revision_events": revision_count, "revision_event_rate": rounded(revision_count / len(annotations)) if annotations else None, "transition_counts": dict(sorted(transitions.items())), "ingestion_delay": timing(ingestion_lags), "revision_delay": timing(revision_lags)}


QUERY_FUNCTIONS: dict[str, Callable[[dict[str, list[dict[str, Any]]], dict[str, Any]], dict[str, Any]]] = {
    f"query{number:02d}": globals()[f"query{number:02d}"] for number in range(16, 31)
}


def run_task(task_id: str, data_dir: Path) -> dict[str, Any]:
    if task_id not in QUERY_FUNCTIONS:
        raise ValueError(f"Unknown advanced query: {task_id}")
    bundle, config = load_bundle(data_dir)
    return QUERY_FUNCTIONS[task_id](bundle, config)


def cli(default_task_id: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", nargs="?" if default_task_id else None, default=default_task_id, choices=sorted(QUERY_FUNCTIONS))
    parser.add_argument("--data-dir", type=Path, default=Path("sample_data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    answer = run_task(args.task_id, args.data_dir.resolve())
    payload = json.dumps(answer, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    cli()
