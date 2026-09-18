#!/usr/bin/env python3
"""Generate one profile, validate it, and run all 30 reference queries."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "simulation"
QUERY_PROJECT = ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "pilot", "simulation"), default="pilot")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> dict:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr[-4000:]}")
    return json.loads(completed.stdout)


def main() -> None:
    args = parse_args()
    profiles = json.loads((ROOT / "config" / "profiles.json").read_text(encoding="utf-8"))
    profile = profiles[args.profile]
    output_root = args.output_root.expanduser().resolve()
    profile_dir = output_root / args.profile
    data_dir = profile_dir / "canonical_data"
    answer_dir = profile_dir / "reference_answers"
    if profile_dir.exists() and any(profile_dir.iterdir()):
        if not args.replace:
            raise SystemExit(f"Profile output exists: {profile_dir}. Add --replace.")
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    generation_command = [
        sys.executable, str(HERE / "generate_synthetic_corpus.py"),
        "--output-dir", str(data_dir),
        "--task-count", str(profile["task_count"]),
        "--model-count", str(profile["model_count"]),
        "--seed", str(profile["seed"]),
        "--bootstrap-repetitions", str(profile["bootstrap_repetitions"]),
        "--permutation-repetitions", str(profile["permutation_repetitions"]),
    ]
    if profile.get("target_trace_count") is not None:
        generation_command.extend(["--target-trace-count", str(profile["target_trace_count"])])
    generation = run(generation_command)
    validation = run([
        sys.executable, str(HERE / "validate_synthetic_corpus.py"),
        "--data-dir", str(data_dir), "--output", str(profile_dir / "validation_report.json"),
    ])
    queries = run([
        sys.executable, str(HERE / "run_timed_queries.py"),
        "--data-dir", str(data_dir), "--query-project", str(QUERY_PROJECT),
        "--output-dir", str(answer_dir),
    ])
    summary = {
        "profile": args.profile,
        "parameters": profile,
        "generation": {"dataset_version": generation["dataset_version"], "counts": generation["counts"]},
        "validation_passed": validation["passed"],
        "query_count": queries["query_count"],
        "query_success": queries["success"],
        "failed_queries": queries["failed"],
        "query_elapsed_seconds": queries["total_elapsed_seconds"],
        "slowest_queries": queries["slowest_queries"],
        "profile_dir": profile_dir.relative_to(ROOT).as_posix() if profile_dir.is_relative_to(ROOT) else str(profile_dir),
    }
    (profile_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
