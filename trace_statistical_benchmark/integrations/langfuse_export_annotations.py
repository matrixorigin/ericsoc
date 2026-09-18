#!/usr/bin/env python3
"""Export UI annotations through Langfuse Scores API v3."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("benchmark_data/simulation/canonical_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("output/manual_annotations.jsonl"))
    parser.add_argument("--raw-output", type=Path, default=Path("output/manual_scores_raw.jsonl"))
    parser.add_argument("--all-sources", action="store_true", help="Include API and EVAL scores too.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_id(*parts: Any) -> str:
    raw = "\x1f".join(json.dumps(part, ensure_ascii=False, sort_keys=True) for part in parts)
    return "ann-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class LangfuseHttpClient:
    def __init__(self) -> None:
        names = ["LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
        missing = [name for name in names if not os.getenv(name)]
        if missing:
            raise SystemExit("Missing environment variables: " + ", ".join(missing))
        self.base_url = str(os.environ["LANGFUSE_BASE_URL"]).rstrip("/")
        auth = base64.b64encode(
            f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
        ).decode()
        self.headers = {"Authorization": f"Basic {auth}", "Accept": "application/json"}

    def get(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(self.base_url + path, headers=self.headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Langfuse returned HTTP {exc.code}: {detail}") from exc


def fetch_scores(client: LangfuseHttpClient, all_sources: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params = {"fields": "details,subject,annotation", "limit": "100"}
        if not all_sources:
            params["source"] = "ANNOTATION"
        if cursor:
            params["cursor"] = cursor
        page = client.get("/api/public/v3/scores?" + urllib.parse.urlencode(params))
        rows.extend(page.get("data") or [])
        cursor = (page.get("meta") or {}).get("cursor")
        if not cursor:
            return rows


def score_target(score: dict[str, Any]) -> tuple[str, str | None, str | None]:
    subject = score.get("subject") or {}
    kind = str(subject.get("kind") or "").lower()
    target_id = subject.get("id")
    trace_id = subject.get("traceId") or subject.get("trace_id")
    if kind == "trace":
        trace_id = target_id
    return kind, str(target_id) if target_id else None, str(trace_id) if trace_id else None


def score_to_annotation(score: dict[str, Any]) -> dict[str, Any] | None:
    target_type, target_id, trace_id = score_target(score)
    if target_type not in {"trace", "observation", "session"} or not target_id:
        return None
    score_name = str(score.get("name") or "unnamed_score")
    return {
        "annotation_id": stable_id("langfuse-score", score.get("id")),
        "target_type": target_type,
        "target_id": target_id,
        "trace_id": trace_id,
        "label_schema_version": "langfuse-score-v1",
        "label_json": {
            "score_name": score_name,
            "score_value": score.get("value"),
            "score_data_type": score.get("dataType"),
        },
        "annotator_id": score.get("authorUserId") or "langfuse-user-unknown",
        "annotator_type": "human" if str(score.get("source", "")).upper() == "ANNOTATION" else "imported",
        "labeled_at": score.get("timestamp") or score.get("createdAt"),
        "ingested_at": score.get("createdAt") or score.get("timestamp"),
        "source": "langfuse-manual-annotation" if str(score.get("source", "")).upper() == "ANNOTATION" else "langfuse-score",
        "supersedes_annotation_id": None,
        "retracted": False,
        "comment": score.get("comment"),
        "metadata": {
            "langfuse_score_id": score.get("id"),
            "langfuse_source": score.get("source"),
            "config_id": score.get("configId"),
            "queue_id": score.get("queueId"),
            "environment": score.get("environment"),
            "project_id": score.get("projectId"),
            "updated_at": score.get("updatedAt"),
            "score_metadata": score.get("metadata") or {},
        },
    }


def main() -> None:
    args = parse_args()
    traces = read_jsonl(args.data_dir / "traces.jsonl")
    observations = read_jsonl(args.data_dir / "observations.jsonl")
    sessions = read_jsonl(args.data_dir / "sessions.jsonl")
    trace_ids = {str(row["trace_id"]) for row in traces}
    observation_ids = {str(row["observation_id"]) for row in observations}
    session_ids = {str(row["session_id"]) for row in sessions}

    scores = fetch_scores(LangfuseHttpClient(), args.all_sources)
    selected_scores: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for score in scores:
        target_type, target_id, trace_id = score_target(score)
        belongs = (
            (target_type == "trace" and target_id in trace_ids)
            or (target_type == "observation" and (target_id in observation_ids or trace_id in trace_ids))
            or (target_type == "session" and target_id in session_ids)
        )
        if not belongs:
            continue
        selected_scores.append(score)
        annotation = score_to_annotation(score)
        if annotation is not None:
            annotations.append(annotation)

    write_jsonl(args.output, annotations)
    write_jsonl(args.raw_output, selected_scores)
    summary = {
        "project_scores_checked": len(scores),
        "simulation_scores_selected": len(selected_scores),
        "annotations_written": len(annotations),
        "score_names": sorted({str(row.get("name")) for row in selected_scores}),
        "authors": sorted({str(row.get("authorUserId") or "unknown") for row in selected_scores}),
        "output": str(args.output),
        "raw_output": str(args.raw_output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not annotations:
        print("No matching annotations found. Add a UI annotation to one uploaded trace, then run this command again.")


if __name__ == "__main__":
    main()
