#!/usr/bin/env python3
"""Check whether all 30 query answers recover the simulation corpus ground truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


TABLES = ("models", "tasks", "sessions", "traces", "observations", "generations", "tool_calls", "model_answers", "annotations")
NOTEBOOK_PATTERN = re.compile(r"hourly_h3_analysis_v\d+\.ipynb", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "benchmark_data/simulation/canonical_data")
    parser.add_argument("--answers-dir", type=Path, default=root / "benchmark_data/simulation/reference_answers")
    parser.add_argument("--rules", type=Path, default=root / "config/acceptance_rules.json")
    parser.add_argument("--output-dir", type=Path, default=root / "benchmark_data/simulation/statistical_validation")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def duration_ms(row: dict[str, Any]) -> int:
    return round((parse_time(row["end_time"]) - parse_time(row["start_time"])).total_seconds() * 1000)


def rounded(value: float, digits: int = 3) -> float:
    return round(value, digits)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir.expanduser().resolve()
    answers_dir = args.answers_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    bundle = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in TABLES}
    answers = {f"query{number:02d}": json.loads((answers_dir / f"query{number:02d}.json").read_text(encoding="utf-8")) for number in range(1, 31)}
    truth = json.loads((data_dir / "private_ground_truth.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((data_dir / "benchmark_config.json").read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []

    def add(query_id: str, family: str, kind: str, rule: str, expected: Any, observed: Any, passed: bool) -> None:
        results.append({
            "query_id": query_id,
            "family": family,
            "validation_kind": kind,
            "rule": rule,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
        })

    traces = bundle["traces"]
    observations = bundle["observations"]
    tool_calls = bundle["tool_calls"]
    generations = bundle["generations"]
    sessions = bundle["sessions"]
    trace_map = {row["trace_id"]: row for row in traces}
    observation_map = {row["observation_id"]: row for row in observations}
    trace_status = Counter(row["status"] for row in traces)
    tool_status = Counter(row["status"] for row in tool_calls)

    q = answers["query01"]
    expected_q01 = {"total_traces": len(traces), "success": trace_status["success"], "error": trace_status["error"], "other": len(traces) - trace_status["success"] - trace_status["error"]}
    add("query01", "descriptive", "exact", "Trace status totals equal the frozen corpus and the declared simulation size.", {**expected_q01, "declared_total": rules["expected_trace_count"]}, q, q == expected_q01 and len(traces) == rules["expected_trace_count"])

    q = answers["query02"]
    expected_ids = sorted(row["trace_id"] for row in traces if row["status"] == "error")
    observed_ids = sorted(row["trace_id"] for row in q["rows"])
    add("query02", "workflow", "exact", "Every failed trace appears once.", len(expected_ids), q["failed_trace_count"], q["failed_trace_count"] == len(expected_ids) and observed_ids == expected_ids)

    q = answers["query03"]
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_session[trace["session_id"]].append(trace)
    expected_latest = {session_id: max(rows, key=lambda row: (row["end_time"], row["trace_id"]))["trace_id"] for session_id, rows in by_session.items()}
    observed_latest = {row["session_id"]: row["trace_id"] for row in q["rows"]}
    add("query03", "relational", "exact", "One correct latest trace is returned per session.", len(sessions), q["session_count"], q["session_count"] == len(sessions) and observed_latest == expected_latest)

    q = answers["query04"]
    maximum = max(row["duration_ms"] for row in traces)
    expected_longest = sorted(row["trace_id"] for row in traces if row["duration_ms"] == maximum)
    observed_longest = sorted(row["trace_id"] for row in q["rows"])
    add("query04", "descriptive", "exact", "Maximum duration and all ties are correct.", {"duration_ms": maximum, "trace_ids": expected_longest}, {"duration_ms": q["longest_duration_ms"], "trace_ids": observed_longest}, q["longest_duration_ms"] == maximum and observed_longest == expected_longest)

    q = answers["query05"]
    expected_tools: dict[str, Counter[str]] = defaultdict(Counter)
    for row in tool_calls:
        expected_tools[row["tool_name"]][row["status"]] += 1
    expected_rows = [{"tool_name": name, "total": sum(counts.values()), "success": counts["success"], "error": counts["error"]} for name, counts in sorted(expected_tools.items())]
    add("query05", "workflow", "exact", "Tool totals match tool-call rows.", len(tool_calls), q["tool_call_count"], q["tool_call_count"] == len(tool_calls) and q["rows"] == expected_rows)

    q = answers["query06"]
    expected_failed_tools = sorted((row["trace_id"], row["tool_name"], row["tool_index"], round(row["duration_ms"])) for row in tool_calls if row["status"] != "success")
    observed_failed_tools = sorted((row["trace_id"], row["tool_name"], row["tool_index"], row["duration_ms"]) for row in q["rows"])
    add("query06", "workflow", "exact", "All failed tool calls are returned.", len(expected_failed_tools), q["failed_tool_call_count"], q["failed_tool_call_count"] == len(expected_failed_tools) and observed_failed_tools == expected_failed_tools)

    q = answers["query07"]
    totals = {row["trace_id"]: int(row["usage"]["input"] or 0) + int(row["usage"]["output"] or 0) for row in traces}
    largest = max(totals.values())
    largest_ids = sorted(trace_id for trace_id, value in totals.items() if value == largest)
    add("query07", "descriptive", "exact", "Largest input-plus-output token total is correct.", {"value": largest, "trace_ids": largest_ids}, {"value": q["largest_input_plus_output"], "trace_ids": q["largest_trace_ids"]}, q["largest_input_plus_output"] == largest and q["largest_trace_ids"] == largest_ids and len(q["rows"]) == len(traces))

    q = answers["query08"]
    expected_generation_counts = Counter(row["trace_id"] for row in generations)
    observed_generation_counts = {row["trace_id"]: row["generation_count"] for row in q["rows"]}
    count_match = all(observed_generation_counts.get(trace_id) == expected_generation_counts[trace_id] for trace_id in trace_map)
    add("query08", "descriptive", "exact", "Generation counts cover every trace.", len(generations), sum(observed_generation_counts.values()), count_match and sum(observed_generation_counts.values()) == len(generations))

    q = answers["query09"]
    expected_top = sorted(tool_calls, key=lambda row: (-row["duration_ms"], row["trace_id"], row["observation_id"]))[:5]
    expected_top_ids = [row["observation_id"] for row in expected_top]
    observed_top_ids = [row["observation_id"] for row in q["rows"]]
    add("query09", "workflow", "exact", "Top five tool calls use deterministic ordering.", expected_top_ids, observed_top_ids, observed_top_ids == expected_top_ids)

    q = answers["query10"]
    expected_profiles = []
    for status in sorted(trace_status):
        rows = [row for row in traces if row["status"] == status]
        expected_profiles.append({
            "status": status,
            "trace_count": len(rows),
            "mean_duration_ms": rounded(mean(row["duration_ms"] for row in rows)),
            "mean_tool_count": rounded(mean(row["metadata"]["tool_count"] for row in rows)),
            "mean_input_tokens": rounded(mean(row["usage"]["input"] for row in rows)),
            "mean_output_tokens": rounded(mean(row["usage"]["output"] for row in rows)),
        })
    add("query10", "comparison", "exact", "Status profiles match direct group summaries.", expected_profiles, q["rows"], q["rows"] == expected_profiles)

    q = answers["query11"]
    recovered = 0
    for failed in (row for row in traces if row["status"] == "error"):
        failed_end = parse_time(failed["end_time"])
        recovered += any(row["session_id"] == failed["session_id"] and row["status"] == "success" and parse_time(row["start_time"]) > failed_end for row in traces)
    add("query11", "workflow", "exact", "Recovery count matches direct session-time search.", recovered, q["failures_with_later_success"], q["failures_with_later_success"] == recovered)

    q = answers["query12"]
    timeline_ids = [attempt["trace_id"] for session in q["sessions"] for attempt in session["attempts"]]
    add("query12", "relational", "exact", "Every trace appears once in the session timelines.", {"sessions": len(sessions), "traces": len(traces)}, {"sessions": q["session_count"], "traces": len(timeline_ids)}, q["session_count"] == len(sessions) and sorted(timeline_ids) == sorted(trace_map))

    q = answers["query13"]
    available = sum(isinstance(row.get("output"), str) and bool(row["output"].strip()) for row in traces)
    expected_output_counts = {"available": available, "missing": len(traces) - available}
    observed_output_counts = {"available": q["available"], "missing": q["missing"]}
    add("query13", "descriptive", "exact", "Output availability matches the trace table.", expected_output_counts, observed_output_counts, expected_output_counts == observed_output_counts)

    q = answers["query14"]
    expected_versions = sorted({row["metadata"]["notebook_version"].lower() for row in bundle["tasks"]})
    add("query14", "relational", "exact", "All generated notebook versions are recovered.", expected_versions, q["all_notebook_versions"], q["all_notebook_versions"] == expected_versions)

    q = answers["query15"]
    integrity_empty = all(not q[field] for field in ("traces_missing_model", "traces_missing_output", "traces_missing_usage", "traces_without_exactly_one_root", "broken_trace_links", "broken_parent_links", "invalid_timestamp_rows") if field not in {"traces_missing_output"})
    add("query15", "relational", "structural", "No broken links, roots, model fields, usage fields, or timestamps.", "all structural issue lists empty", {key: len(q[key]) for key in ("traces_missing_model", "traces_missing_usage", "traces_without_exactly_one_root", "broken_trace_links", "broken_parent_links", "invalid_timestamp_rows")}, integrity_empty and q["trace_count"] == len(traces) and q["observation_count"] == len(observations))

    q = answers["query16"]
    add("query16", "relational", "structural", "Every model answer has complete, unambiguous lineage.", {"answers": len(bundle["model_answers"]), "issues": {}}, {"answers": q["answer_count"], "issues": q["issue_counts"]}, q["answer_count"] == len(bundle["model_answers"]) and not q["issue_counts"])

    q = answers["query17"]
    model_a, model_b = config["selected_models"][:2]
    expected_a = truth["models"][model_a]["final_answer_accuracy"]
    expected_b = truth["models"][model_b]["final_answer_accuracy"]
    observed_gap = q["accuracy_a"] - q["accuracy_b"]
    expected_gap = expected_a - expected_b
    passed = q["matched_tasks"] == rules["expected_task_count"] and abs(q["accuracy_a"] - expected_a) < 1e-9 and abs(q["accuracy_b"] - expected_b) < 1e-9 and abs(observed_gap) >= rules["minimum_model_accuracy_gap"] and q["exact_mcnemar_p_value"] < 0.05
    add("query17", "comparison", "known-effect", "Recover the selected models' paired accuracy difference.", {"accuracy_a": expected_a, "accuracy_b": expected_b, "gap": expected_gap, "p<": 0.05}, {"accuracy_a": q["accuracy_a"], "accuracy_b": q["accuracy_b"], "gap": observed_gap, "p": q["exact_mcnemar_p_value"]}, passed)

    q = answers["query18"]
    lower, upper = q["bootstrap_95_ci"]
    add("query18", "comparison", "known-effect", "The slower selected model has a positive paired latency gap with an interval above zero.", "median gap > 0 and CI lower > 0", {"matched": q["matched_successful_tasks"], "median_gap_ms": q["median_difference_a_minus_b_ms"], "ci": q["bootstrap_95_ci"]}, q["matched_successful_tasks"] >= rules["expected_task_count"] * 0.75 and q["median_difference_a_minus_b_ms"] > 0 and lower > 0 and upper > lower)

    q = answers["query19"]
    all_model_ids = sorted(row["model_id"] for row in bundle["models"])
    add("query19", "comparison", "known-null", "Configured quality, latency, and cost tradeoffs keep both systems on the Pareto frontier.", all_model_ids, sorted(q["frontier"]), q["common_task_count"] == rules["expected_task_count"] and sorted(q["frontier"]) == all_model_ids)

    q = answers["query20"]
    ci = q["bootstrap_95_ci"]
    interaction = q["interaction_hard_minus_easy"]
    add("query20", "comparison", "known-null", "No model-specific difficulty interaction was programmed.", f"absolute interaction <= {rules['null_interaction_absolute_tolerance']} and CI contains 0", {"interaction": interaction, "ci": ci}, abs(interaction) <= rules["null_interaction_absolute_tolerance"] and ci[0] <= 0 <= ci[1])

    q = answers["query21"]
    add("query21", "inference", "known-effect", "Higher generated complexity is negatively associated with correctness.", "rho < 0 and permutation p < 0.05", {"n": q["n"], "rho": q["spearman_rho"], "p": q["permutation_p_value"]}, q["n"] == rules["expected_task_count"] * rules["expected_model_count"] and q["spearman_rho"] < 0 and q["permutation_p_value"] < 0.05)

    q = answers["query22"]
    cells = list(q["table"].values())
    add("query22", "inference", "known-effect", "Tool failure is associated with final trace failure but is not deterministic.", f"all cells > 0, risk ratio > {rules['minimum_tool_failure_risk_ratio']}, CI lower > 1", {"table": q["table"], "risk_ratio": q["risk_ratio"], "ci": q["risk_ratio_95_ci"], "p": q["fisher_exact_p_value"]}, all(value > 0 for value in cells) and q["risk_ratio"] > rules["minimum_tool_failure_risk_ratio"] and q["risk_ratio_95_ci"][0] > 1 and q["fisher_exact_p_value"] < 0.05)

    q = answers["query23"]
    recovery_rates = [row["recovery_rate"] for row in q["strata"]]
    add("query23", "inference", "known-effect", "Recovery rises across the generated retry-count levels.", "at least 3 levels, nondecreasing rates, positive z, p < 0.05", {"retry_levels": [row["retry_count"] for row in q["strata"]], "rates": recovery_rates, "z": q["cochran_armitage_z"], "p": q["trend_p_value"]}, len(recovery_rates) >= 3 and recovery_rates == sorted(recovery_rates) and q["cochran_armitage_z"] > 0 and q["trend_p_value"] < 0.05)

    q = answers["query24"]
    uniform_hhi = 1 / len(q["tools"])
    add("query24", "inference", "known-null", "Failures were spread approximately uniformly across five tools.", {"hhi": uniform_hhi, "tolerance": rules["uniform_tool_hhi_tolerance"], "gini<": 0.05}, {"hhi": q["hhi"], "gini": q["gini"], "failed_tools": q["failed_tool_calls"]}, len(q["tools"]) == 5 and abs(q["hhi"] - uniform_hhi) <= rules["uniform_tool_hhi_tolerance"] and q["gini"] < 0.05)

    q = answers["query25"]
    true_index = truth["drift"]["first_drifted_trace_index_one_based"]
    selected_index = q["selected_change_point"]["split_index"]
    location_error = abs(selected_index - true_index) / len(traces)
    add("query25", "inference", "location-tolerance", "Detect the programmed latency drift within the allowed trace-position tolerance.", {"true_index": true_index, "max_error_fraction": rules["drift_location_tolerance_fraction"]}, {"selected_index": selected_index, "error_fraction": round(location_error, 6), "metric": q["selected_change_point"]["metric"]}, q["evidence_after_correction"] and q["selected_change_point"]["metric"] == "mean_latency_ms" and location_error <= rules["drift_location_tolerance_fraction"])

    q = answers["query26"]
    add("query26", "annotation", "structural", "All generated annotation events satisfy the canonical contract.", {"annotations": len(bundle["annotations"]), "invalid": 0}, {"annotations": q["annotation_count"], "invalid": q["invalid_annotation_count"]}, q["annotation_count"] == len(bundle["annotations"]) and q["invalid_annotation_count"] == 0)

    q = answers["query27"]
    expected_effective = rules["expected_task_count"] * 2
    add("query27", "annotation", "exact", "One effective label remains per task and simulated reviewer.", expected_effective, q["effective_label_count"], q["effective_label_count"] == expected_effective and not q["history_anomalies"])

    q = answers["query28"]
    pair = q["annotator_pairs"][0]
    add("query28", "annotation", "range", "Simulated reviewer agreement is neither random nor perfect.", {"shared": rules["expected_task_count"], "raw_agreement_range": [rules["minimum_annotation_raw_agreement"], rules["maximum_annotation_raw_agreement"]], "kappa_range": [0.2, 0.6]}, {"shared": pair["shared_targets"], "raw_agreement": pair["raw_agreement"], "kappa": pair["cohen_kappa"]}, pair["shared_targets"] == rules["expected_task_count"] and rules["minimum_annotation_raw_agreement"] <= pair["raw_agreement"] <= rules["maximum_annotation_raw_agreement"] and 0.2 <= pair["cohen_kappa"] <= 0.6)

    q = answers["query29"]
    pair = q["pairs"][0]
    add("query29", "annotation", "known-null", "No strong reviewer positive-rate bias should be detected.", "absolute paired difference < 0.05 and adjusted p >= 0.05", {"difference": pair["paired_difference_a_minus_b"], "adjusted_p": pair["holm_adjusted_p_value"]}, abs(pair["paired_difference_a_minus_b"]) < 0.05 and pair["holm_adjusted_p_value"] >= 0.05)

    q = answers["query30"]
    rate_error = abs(q["revision_event_rate"] - rules["annotation_revision_rate_target"])
    add("query30", "annotation", "range", "Revision rate recovers the programmed 8% process and all timing values are nonnegative.", {"target_rate": rules["annotation_revision_rate_target"], "tolerance": rules["annotation_revision_rate_tolerance"]}, {"rate": q["revision_event_rate"], "events": q["annotation_events"], "revision_events": q["revision_events"]}, q["annotation_events"] == len(bundle["annotations"]) and rate_error <= rules["annotation_revision_rate_tolerance"] and q["ingestion_delay"]["median_seconds"] >= 0 and q["revision_delay"]["median_seconds"] >= 0)

    failed = [row["query_id"] for row in results if not row["passed"]]
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    kind_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        family_counts[row["family"]]["total"] += 1
        family_counts[row["family"]]["passed"] += int(row["passed"])
        kind_counts[row["validation_kind"]]["total"] += 1
        kind_counts[row["validation_kind"]]["passed"] += int(row["passed"])

    report = {
        "validation_version": rules["validation_version"],
        "dataset_version": manifest["dataset_version"],
        "dataset_seed": manifest["seed"],
        "query_count": len(results),
        "passed_count": len(results) - len(failed),
        "failed_count": len(failed),
        "failed_queries": failed,
        "all_passed": not failed,
        "family_summary": {key: dict(value) for key, value in sorted(family_counts.items())},
        "validation_kind_summary": {key: dict(value) for key, value in sorted(kind_counts.items())},
        "results": results,
    }
    (output_dir / "truth_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "query_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("query_id", "family", "validation_kind", "passed", "rule", "expected", "observed"))
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "expected": json.dumps(row["expected"], ensure_ascii=False, sort_keys=True), "observed": json.dumps(row["observed"], ensure_ascii=False, sort_keys=True)})

    snapshot_files = [data_dir / "manifest.json", data_dir / "benchmark_config.json", data_dir / "private_ground_truth.json"]
    snapshot_files.extend(data_dir / f"{name}.jsonl" for name in TABLES)
    answer_files = [answers_dir / f"query{number:02d}.json" for number in range(1, 31)]
    snapshot = {
        "freeze_version": "simulation-snapshot-v1",
        "dataset_version": manifest["dataset_version"],
        "seed": manifest["seed"],
        "source_data_directory": portable_path(data_dir, project_root),
        "source_answers_directory": portable_path(answers_dir, project_root),
        "counts": manifest["counts"],
        "data_checksums_sha256": {path.name: sha256(path) for path in snapshot_files},
        "answer_checksums_sha256": {path.name: sha256(path) for path in answer_files},
        "truth_validation_report_sha256": sha256(output_dir / "truth_validation_report.json"),
    }
    (output_dir / "frozen_snapshot_manifest.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("validation_version", "dataset_version", "query_count", "passed_count", "failed_count", "failed_queries", "all_passed", "family_summary", "validation_kind_summary")}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
