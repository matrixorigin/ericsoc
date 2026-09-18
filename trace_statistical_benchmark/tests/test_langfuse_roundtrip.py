from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS = ROOT / "integrations"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class RoundTripToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upload = load_module("langfuse_upload", INTEGRATIONS / "langfuse_upload.py")
        cls.export = load_module("langfuse_export", INTEGRATIONS / "langfuse_export_annotations.py")
        cls.verify = load_module("langfuse_verify", INTEGRATIONS / "langfuse_verify_inventory.py")

    def test_ingestion_event_ids_are_stable(self) -> None:
        trace = {
            "trace_id": "trace-one",
            "session_id": "session-one",
            "task_id": "task-one",
            "model_id": "model-one",
            "model": "Astra",
            "name": "test-trace",
            "status": "success",
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T00:00:01Z",
            "duration_ms": 1000,
            "input": "input",
            "output": "output",
            "attempt_number": 1,
            "usage": {"input": 10, "output": 2},
            "metadata": {},
        }
        observations = [
            {
                "observation_id": "obs-one",
                "trace_id": "trace-one",
                "parent_observation_id": None,
                "type": "generation",
                "name": "llm-round",
                "status": "success",
                "start_time": "2026-01-01T00:00:00Z",
                "end_time": "2026-01-01T00:00:01Z",
                "input": "prompt",
                "output": "answer",
                "model": "Astra",
                "usage": {"input": 10, "output": 2},
                "metadata": {},
            }
        ]
        first = self.upload.build_events(trace, observations, "test-env", "dataset-v1")
        second = self.upload.build_events(trace, observations, "test-env", "dataset-v1")
        self.assertEqual(len(first), 2)
        self.assertEqual([event.id for event in first], [event.id for event in second])

    def test_score_conversion_keeps_subject_author_and_times(self) -> None:
        score = {
            "id": "score-one",
            "name": "Answer_Correct",
            "value": True,
            "dataType": "BOOLEAN",
            "source": "ANNOTATION",
            "timestamp": "2026-09-15T12:00:00Z",
            "createdAt": "2026-09-15T12:00:02Z",
            "updatedAt": "2026-09-15T12:00:03Z",
            "authorUserId": "user-one",
            "comment": "checked",
            "subject": {"kind": "observation", "id": "obs-one", "traceId": "trace-one"},
        }
        row = self.export.score_to_annotation(score)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["target_type"], "observation")
        self.assertEqual(row["target_id"], "obs-one")
        self.assertEqual(row["trace_id"], "trace-one")
        self.assertEqual(row["annotator_id"], "user-one")
        self.assertEqual(row["labeled_at"], "2026-09-15T12:00:00Z")
        self.assertEqual(row["ingested_at"], "2026-09-15T12:00:02Z")

    def test_observation_reader_falls_back_to_v1(self) -> None:
        class Item:
            def __init__(self, item_id: str) -> None:
                self.item_id = item_id

            def model_dump(self, **_kwargs):
                return {"id": self.item_id, "trace_id": "trace-one", "type": "SPAN"}

        class V2:
            @staticmethod
            def get_many(**_kwargs):
                raise RuntimeError("The observations v2 API is only available in a Langfuse v4 write mode")

        class V1:
            @staticmethod
            def get_many(*, page: int, **_kwargs):
                return SimpleNamespace(
                    data=[Item(f"obs-{page}")],
                    meta=SimpleNamespace(page=page, total_items=2, total_pages=2),
                )

        client = SimpleNamespace(
            api=SimpleNamespace(
                observations=V2(),
                legacy=SimpleNamespace(observations_v1=V1()),
            )
        )
        rows, mode = self.verify.fetch_observations(client, "test-env", 100)
        self.assertEqual(mode, "legacy-observations-v1")
        self.assertEqual([row["id"] for row in rows], ["obs-1", "obs-2"])

    def test_trace_reader_collects_all_pages(self) -> None:
        class TraceApi:
            @staticmethod
            def list(*, page: int, **_kwargs):
                return SimpleNamespace(
                    data=[SimpleNamespace(id=f"trace-{page}")],
                    meta=SimpleNamespace(page=page, total_items=3, total_pages=3),
                )

        client = SimpleNamespace(api=SimpleNamespace(trace=TraceApi()))
        trace_ids = self.verify.fetch_trace_ids(client, "test-env", workers=2)
        self.assertEqual(trace_ids, {"trace-1", "trace-2", "trace-3"})

    def test_local_annotation_queries_build_joined_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            output_dir = root / "analysis"
            annotation_path = root / "annotations.jsonl"
            write_jsonl(
                data_dir / "traces.jsonl",
                [{"trace_id": "trace-one", "model": "Astra", "status": "success"}],
            )
            write_jsonl(
                data_dir / "observations.jsonl",
                [{"observation_id": "obs-one", "trace_id": "trace-one", "type": "generation"}],
            )
            write_jsonl(
                data_dir / "model_answers.jsonl",
                [{"trace_id": "trace-one", "correct": True, "answer_id": "answer-one"}],
            )
            write_jsonl(
                annotation_path,
                [
                    {
                        "annotation_id": "ann-one",
                        "target_type": "trace",
                        "target_id": "trace-one",
                        "trace_id": "trace-one",
                        "label_json": {"score_name": "Answer_Correct", "score_value": True},
                        "annotator_id": "user-one",
                        "labeled_at": "2026-09-15T12:00:00Z",
                        "ingested_at": "2026-09-15T12:00:02Z",
                    }
                ],
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(INTEGRATIONS / "analyze_annotations.py"),
                    "--data-dir",
                    str(data_dir),
                    "--annotations",
                    str(annotation_path),
                    "--output-dir",
                    str(output_dir),
                    "--expect-min",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output_dir / "annotation_query_results.json").read_text())
            self.assertTrue(report["all_links_valid"])
            self.assertEqual(report["queries"]["query_a06_answer_correct_agreement"]["agreement_rate"], 1.0)
            bundles = (output_dir / "annotated_trace_bundles.jsonl").read_text().splitlines()
            self.assertEqual(len(bundles), 1)


if __name__ == "__main__":
    unittest.main()
