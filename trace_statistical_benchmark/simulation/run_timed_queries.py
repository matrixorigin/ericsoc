#!/usr/bin/env python3
"""Run all reference queries and record per-query wall-clock time."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--query-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    project = args.query_project.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rows = []
    total_start = time.perf_counter()
    for number in range(1, 31):
        query_id = f"query{number:02d}"
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-m", f"queries.{query_id}", "--data-dir", str(data_dir)],
            cwd=project,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        row = {"query_id": query_id, "return_code": completed.returncode, "elapsed_seconds": round(elapsed, 6)}
        if completed.returncode == 0:
            answer = json.loads(completed.stdout)
            (output_dir / f"{query_id}.json").write_text(
                json.dumps(answer, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            row["status"] = "success"
        else:
            row["status"] = "error"
            row["error"] = completed.stderr[-2000:]
        rows.append(row)

    total_elapsed = time.perf_counter() - total_start
    with (output_dir / "query_run_summary.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    failed = [row["query_id"] for row in rows if row["status"] != "success"]
    slowest = sorted(rows, key=lambda row: (-row["elapsed_seconds"], row["query_id"]))[:5]
    report = {
        "query_count": len(rows),
        "success": len(rows) - len(failed),
        "failed": failed,
        "total_elapsed_seconds": round(total_elapsed, 6),
        "slowest_queries": slowest,
        "output_dir": output_dir.relative_to(project).as_posix() if output_dir.is_relative_to(project) else str(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
