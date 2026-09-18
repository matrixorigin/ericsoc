#!/usr/bin/env python3
"""Validate the machine-readable 30-query benchmark design."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "query_catalog.json"
sys.path.insert(0, str(ROOT))

from reference_queries import QUERY_FUNCTIONS  # noqa: E402


EXPECTED_FAMILIES = {
    "descriptive_distribution",
    "relational_integrity",
    "workflow_failure",
    "comparative_performance",
    "inferential_robustness",
    "annotation_provenance",
}
DIMENSION_KEYS = {
    "relational_depth",
    "statistical_depth",
    "data_complexity",
    "temporal_depth",
}


def validate_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    queries = catalog.get("queries")
    if not isinstance(queries, list):
        raise ValueError("queries must be a list")

    expected_ids = {f"query{number:02d}" for number in range(1, 31)}
    query_ids = [query.get("id") for query in queries]
    if len(queries) != 30 or set(query_ids) != expected_ids or len(set(query_ids)) != 30:
        raise ValueError("catalog must contain each query01-query30 exactly once")

    family_counts = Counter(query.get("family") for query in queries)
    if set(family_counts) != EXPECTED_FAMILIES or set(family_counts.values()) != {5}:
        raise ValueError("catalog must contain six declared families with five queries each")

    for query in queries:
        missing = {
            "id", "family", "title", "question", "status",
            "dimensions", "oracle", "answer_shape",
        } - set(query)
        if missing:
            raise ValueError(f"{query.get('id')} is missing fields: {sorted(missing)}")
        if query["status"] not in {"implemented", "specified"}:
            raise ValueError(f"{query['id']} has an invalid status")
        dimensions = query["dimensions"]
        if set(dimensions) != DIMENSION_KEYS:
            raise ValueError(f"{query['id']} has invalid difficulty dimensions")
        if any(not isinstance(value, int) or not 1 <= value <= 5 for value in dimensions.values()):
            raise ValueError(f"{query['id']} dimension values must be integers from 1 to 5")

    implemented = {query["id"] for query in queries if query["status"] == "implemented"}
    if implemented != set(QUERY_FUNCTIONS):
        raise ValueError("implemented design entries must match executable reference queries")

    return {
        "query_count": len(queries),
        "family_counts": dict(sorted(family_counts.items())),
        "implemented_count": len(implemented),
        "specified_count": len(queries) - len(implemented),
    }


def main() -> None:
    print(json.dumps(validate_catalog(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
