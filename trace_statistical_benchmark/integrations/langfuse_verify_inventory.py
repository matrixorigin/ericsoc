#!/usr/bin/env python3
"""Compare Langfuse records with a local canonical trace corpus."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langfuse import Langfuse


DEFAULT_DATA_DIR = Path("benchmark_data/simulation/canonical_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--environment", default="synthetic-simulation-v1")
    parser.add_argument("--output", type=Path, default=Path("output/remote_inventory_report.json"))
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_environment() -> tuple[str, str, str]:
    names = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing))
    return tuple(str(os.environ[name]).rstrip("/") for name in names)  # type: ignore[return-value]


def fetch_observations_v2(client: Langfuse, environment: str, page_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        page = client.api.observations.get_many(
            environment=environment,
            limit=page_size,
            cursor=cursor,
        )
        rows.extend(item.model_dump(mode="json", by_alias=False) for item in page.data)
        cursor = page.meta.cursor
        if not cursor:
            return rows


def fetch_v1_page(client: Langfuse, environment: str, page_size: int, page_number: int) -> Any:
    for attempt in range(1, 5):
        try:
            return client.api.legacy.observations_v1.get_many(
                environment=environment,
                limit=page_size,
                page=page_number,
            )
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError("Unreachable retry state")


def fetch_trace_page(client: Langfuse, environment: str, page_number: int) -> Any:
    for attempt in range(1, 5):
        try:
            return client.api.trace.list(
                environment=environment,
                limit=100,
                page=page_number,
            )
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError("Unreachable retry state")


def fetch_trace_ids(client: Langfuse, environment: str, workers: int) -> set[str]:
    first_page = fetch_trace_page(client, environment, 1)
    trace_ids = {str(item.id) for item in first_page.data}
    total_items = first_page.meta.total_items
    total_pages = first_page.meta.total_pages
    print(f"Langfuse v3 reports {total_items:,} traces across {total_pages:,} pages.")
    if total_pages <= 1:
        return trace_ids
    completed_pages = 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_trace_page, client, environment, page_number): page_number
            for page_number in range(2, total_pages + 1)
        }
        for future in as_completed(futures):
            page = future.result()
            trace_ids.update(str(item.id) for item in page.data)
            completed_pages += 1
            if completed_pages % 25 == 0 or completed_pages == total_pages:
                print(f"Read {len(trace_ids):,}/{total_items:,} remote traces...")
    return trace_ids


def fetch_observations_v1(
    client: Langfuse,
    environment: str,
    page_size: int,
    workers: int,
) -> list[dict[str, Any]]:
    page_size = min(page_size, 100)
    first_page = fetch_v1_page(client, environment, page_size, 1)
    rows = [item.model_dump(mode="json", by_alias=False) for item in first_page.data]
    total_items = first_page.meta.total_items
    total_pages = first_page.meta.total_pages
    print(f"Langfuse v3 reports {total_items:,} observations across {total_pages:,} pages.")
    if total_pages <= 1:
        return rows

    completed_pages = 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_v1_page, client, environment, page_size, page_number): page_number
            for page_number in range(2, total_pages + 1)
        }
        for future in as_completed(futures):
            page = future.result()
            rows.extend(item.model_dump(mode="json", by_alias=False) for item in page.data)
            completed_pages += 1
            if completed_pages % 50 == 0 or completed_pages == total_pages:
                print(f"Read {len(rows):,}/{total_items:,} remote observations...")
    return rows


def fetch_observations(
    client: Langfuse,
    environment: str,
    page_size: int,
    workers: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    try:
        return fetch_observations_v2(client, environment, page_size), "observations-v2"
    except Exception as exc:
        message = str(exc)
        if "v4 write mode" not in message and "status_code: 404" not in message:
            raise
        print("Observations v2 is unavailable on this Langfuse v3 instance; using legacy v1 pagination.")
        return fetch_observations_v1(client, environment, page_size, workers), "legacy-observations-v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.page_size < 1 or args.wait_seconds < 0 or args.workers < 1:
        raise SystemExit("--page-size and --workers must be positive; --wait-seconds cannot be negative.")
    public_key, secret_key, base_url = require_environment()
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        tracing_enabled=False,
        timeout=90,
    )
    if not client.auth_check():
        raise SystemExit("Langfuse authentication failed.")
    if args.wait_seconds > 0:
        print(f"Waiting {args.wait_seconds} seconds for asynchronous ingestion...")
        time.sleep(args.wait_seconds)

    local_traces = read_jsonl(args.data_dir / "traces.jsonl")
    local_observations = read_jsonl(args.data_dir / "observations.jsonl")
    remote_observations, api_mode = fetch_observations(
        client,
        args.environment,
        args.page_size,
        args.workers,
    )
    remote_trace_ids = fetch_trace_ids(client, args.environment, args.workers)

    local_trace_ids = {str(row["trace_id"]) for row in local_traces}
    local_observation_ids = {str(row["observation_id"]) for row in local_observations}
    remote_observation_ids = {str(row["id"]) for row in remote_observations}
    missing_trace_ids = sorted(local_trace_ids - remote_trace_ids)
    missing_observation_ids = sorted(local_observation_ids - remote_observation_ids)
    unexpected_trace_ids = sorted(remote_trace_ids - local_trace_ids)
    unexpected_observation_ids = sorted(remote_observation_ids - local_observation_ids)

    report = {
        "environment": args.environment,
        "read_api": api_mode,
        "local": {
            "traces": len(local_trace_ids),
            "observations": len(local_observation_ids),
        },
        "remote": {
            "traces": len(remote_trace_ids),
            "observations": len(remote_observation_ids),
            "observations_with_trace_id_in_v1_response": sum(bool(row.get("trace_id")) for row in remote_observations),
            "observation_types": dict(sorted(Counter(str(row.get("type")) for row in remote_observations).items())),
        },
        "differences": {
            "missing_trace_count": len(missing_trace_ids),
            "missing_trace_examples": missing_trace_ids[:20],
            "missing_observation_count": len(missing_observation_ids),
            "missing_observation_examples": missing_observation_ids[:20],
            "unexpected_trace_count": len(unexpected_trace_ids),
            "unexpected_trace_examples": unexpected_trace_ids[:20],
            "unexpected_observation_count": len(unexpected_observation_ids),
            "unexpected_observation_examples": unexpected_observation_ids[:20],
        },
        "exact_match": not any(
            [missing_trace_ids, missing_observation_ids, unexpected_trace_ids, unexpected_observation_ids]
        ),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
