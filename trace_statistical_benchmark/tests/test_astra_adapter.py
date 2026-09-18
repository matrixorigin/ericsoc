from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters" / "astra_session_adapter.py"


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class AstraLogAdapterTest(unittest.TestCase):
    def test_adapter_builds_success_error_and_tool_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            astra_home = root / ".astra"
            source = astra_home / "sessions" / "session-a.jsonl"
            write_rows(source, [
                {"type": "session_start", "ts": "2026-01-01T00:00:00Z", "session_id": "session-a"},
                {
                    "type": "llm_round", "ts": "2026-01-01T00:00:02Z", "turn": 1,
                    "agentic_step": 1, "tokens_in": 10, "tokens_out": 4, "duration_ms": 1000,
                    "tool_calls": [{
                        "tool_call_id": "tool-1", "name": "read_file", "ok": True, "ms": 50,
                        "args_preview": "api_key=secret-value", "result_preview": "done",
                    }],
                    "metadata": {"finish_reason": "tool_calls", "tool_call_names": ["read_file"]},
                },
                {
                    "type": "turn", "ts": "2026-01-01T00:00:03Z", "session_id": "session-a",
                    "turn": 1, "model": "test-model", "user_input": "Read a file",
                    "assistant_output": "Done", "duration_ms": 3000, "tokens_in": 20,
                    "tokens_out": 5, "tool_count": 1,
                },
                {
                    "type": "turn_error", "ts": "2026-01-01T00:00:06Z", "session_id": "session-a",
                    "turn": 2, "model": "test-model", "user_input": "Try again",
                    "duration_ms": 2000, "error": "network failed",
                },
            ])
            output = root / "canonical"
            subprocess.run([
                sys.executable, str(ADAPTER), "--astra-home", str(astra_home),
                "--output-dir", str(output), "--max-traces", "10", "--replace",
            ], check=True, capture_output=True, text=True)

            traces = read_rows(output / "traces.jsonl")
            observations = read_rows(output / "observations.jsonl")
            self.assertEqual([row["status"] for row in traces], ["success", "error"])
            self.assertEqual(len({row["trace_id"] for row in traces}), 2)
            self.assertTrue(any(row["type"] == "generation" for row in observations))
            tool = next(row for row in observations if row["type"] == "tool")
            self.assertIn("[REDACTED]", tool["input"])
            self.assertNotIn("secret-value", tool["input"])


if __name__ == "__main__":
    unittest.main()
