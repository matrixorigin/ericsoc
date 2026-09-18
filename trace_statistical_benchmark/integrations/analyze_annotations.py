#!/usr/bin/env python3
"""Run reproducible local queries over exported Langfuse annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_DATA_DIR = Path("benchmark_data/simulation/canonical_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--annotations", type=Path, default=Path("output/manual_annotations.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/annotation_analysis"))
    parser.add_argument("--expect-min", type=int, default=1)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_name(annotation: dict[str, Any]) -> str:
    return str((annotation.get("label_json") or {}).get("score_name") or "unnamed_score")


def score_value(annotation: dict[str, Any]) -> Any:
    return (annotation.get("label_json") or {}).get("score_value")


def main() -> None:
    args = parse_args()
    annotations = read_jsonl(args.annotations)
    if len(annotations) < args.expect_min:
        raise SystemExit(f"Expected at least {args.expect_min} annotations, found {len(annotations)}.")

    traces = read_jsonl(args.data_dir / "traces.jsonl")
    observations = read_jsonl(args.data_dir / "observations.jsonl")
    answers = read_jsonl(args.data_dir / "model_answers.jsonl")
    trace_map = {str(row["trace_id"]): row for row in traces}
    observation_map = {str(row["observation_id"]): row for row in observations}
    answer_map = {str(row["trace_id"]): row for row in answers}

    invalid_links: list[str] = []
    timing_seconds: list[float] = []
    target_trace_ids: set[str] = set()
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        target_type = str(row.get("target_type"))
        target_id = str(row.get("target_id"))
        trace_id = row.get("trace_id")
        if target_type == "trace":
            trace_id = target_id
            if target_id not in trace_map:
                invalid_links.append(str(row.get("annotation_id")))
        elif target_type == "observation":
            if target_id not in observation_map:
                invalid_links.append(str(row.get("annotation_id")))
            elif not trace_id:
                trace_id = observation_map[target_id]["trace_id"]
        target_trace_ids.add(str(trace_id)) if trace_id else None
        key = (target_type, target_id, score_name(row), str(row.get("annotator_id")))
        grouped[key].append(row)
        labeled = parse_time(row.get("labeled_at"))
        ingested = parse_time(row.get("ingested_at"))
        if labeled and ingested:
            timing_seconds.append((ingested - labeled).total_seconds())

    latest_rows: list[dict[str, Any]] = []
    duplicate_groups = 0
    revision_events = 0
    for rows in grouped.values():
        rows.sort(key=lambda row: (parse_time(row.get("labeled_at")) or datetime.min.replace(tzinfo=timezone.utc), str(row.get("annotation_id"))))
        latest_rows.append(rows[-1])
        if len(rows) > 1:
            duplicate_groups += 1
            revision_events += len(rows) - 1

    distribution = Counter((score_name(row), json.dumps(score_value(row), ensure_ascii=False, sort_keys=True)) for row in latest_rows)
    score_rows = [
        {"score_name": name, "score_value": value, "count": count}
        for (name, value), count in sorted(distribution.items())
    ]
    author_rows = [
        {"annotator_id": author, "annotation_count": count}
        for author, count in sorted(Counter(str(row.get("annotator_id")) for row in latest_rows).items())
    ]

    correctness_rows = [
        row
        for row in latest_rows
        if score_name(row).casefold() == "answer_correct" and str(row.get("trace_id") or row.get("target_id")) in answer_map
    ]
    correctness_match = 0
    for row in correctness_rows:
        trace_id = str(row.get("trace_id") or row.get("target_id"))
        correctness_match += score_value(row) is bool(answer_map[trace_id].get("correct"))

    timing = {
        "count": len(timing_seconds),
        "minimum_seconds": min(timing_seconds) if timing_seconds else None,
        "median_seconds": median(timing_seconds) if timing_seconds else None,
        "mean_seconds": mean(timing_seconds) if timing_seconds else None,
        "maximum_seconds": max(timing_seconds) if timing_seconds else None,
        "negative_count": sum(value < 0 for value in timing_seconds),
    }
    query_results = {
        "query_a01_annotation_inventory": {
            "annotation_events": len(annotations),
            "effective_annotations": len(latest_rows),
            "annotated_traces": len(target_trace_ids),
            "total_local_traces": len(trace_map),
            "trace_coverage": len(target_trace_ids) / len(trace_map) if trace_map else None,
        },
        "query_a02_target_levels": dict(sorted(Counter(str(row.get("target_type")) for row in latest_rows).items())),
        "query_a03_score_distribution": score_rows,
        "query_a04_annotations_by_author": author_rows,
        "query_a05_link_integrity": {"invalid_count": len(invalid_links), "invalid_ids": invalid_links[:20]},
        "query_a06_answer_correct_agreement": {
            "compared": len(correctness_rows),
            "matched": correctness_match,
            "agreement_rate": correctness_match / len(correctness_rows) if correctness_rows else None,
        },
        "query_a07_repeated_labels": {
            "repeated_target_score_author_groups": duplicate_groups,
            "additional_annotation_events": revision_events,
        },
        "query_a08_annotation_timing": timing,
    }
    summary = {
        "all_links_valid": not invalid_links,
        "query_count": len(query_results),
        "queries": query_results,
    }
    write_json(args.output_dir / "annotation_query_results.json", summary)
    write_csv(args.output_dir / "score_distribution.csv", score_rows, ["score_name", "score_value", "count"])
    write_csv(args.output_dir / "annotations_by_author.csv", author_rows, ["annotator_id", "annotation_count"])

    bundles: list[dict[str, Any]] = []
    annotations_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in latest_rows:
        trace_id = row.get("trace_id") or (row.get("target_id") if row.get("target_type") == "trace" else None)
        if trace_id:
            annotations_by_trace[str(trace_id)].append(row)
    observations_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if str(row["trace_id"]) in annotations_by_trace:
            observations_by_trace[str(row["trace_id"])].append(row)
    for trace_id in sorted(annotations_by_trace):
        bundles.append(
            {
                "trace": trace_map.get(trace_id),
                "observations": observations_by_trace.get(trace_id, []),
                "reference_answer": answer_map.get(trace_id),
                "effective_manual_annotations": annotations_by_trace[trace_id],
            }
        )
    bundle_path = args.output_dir / "annotated_trace_bundles.jsonl"
    with bundle_path.open("w", encoding="utf-8") as handle:
        for row in bundles:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Annotated trace bundles: {bundle_path}")
    if invalid_links:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
