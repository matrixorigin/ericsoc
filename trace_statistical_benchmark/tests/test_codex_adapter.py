from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ADAPTERS = Path(__file__).resolve().parents[1] / "adapters"
sys.path.insert(0, str(ADAPTERS))

from codex_rollout_adapter import convert_rollouts  # noqa: E402


class CodexAdapterTest(unittest.TestCase):
    def test_primary_turn_becomes_one_trace(self) -> None:
        rows = [
            {"timestamp": "2026-01-01T00:00:00Z", "type": "session_meta", "payload": {"id": "session-1", "source": "vscode", "thread_source": "user", "model_provider": "openai"}},
            {"timestamp": "2026-01-01T00:00:01Z", "type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-1"}},
            {"timestamp": "2026-01-01T00:00:02Z", "type": "turn_context", "payload": {"turn_id": "turn-1", "model": "test-model", "effort": "high"}},
            {"timestamp": "2026-01-01T00:00:03Z", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Run the task"}]}},
            {"timestamp": "2026-01-01T00:00:04Z", "type": "response_item", "payload": {"type": "custom_tool_call", "call_id": "call-1", "name": "exec", "input": "pwd"}},
            {"timestamp": "2026-01-01T00:00:05Z", "type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": "call-1", "output": [{"type": "input_text", "text": "ok"}]}},
            {"timestamp": "2026-01-01T00:00:06Z", "type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": {"input_tokens": 10, "output_tokens": 3, "cached_input_tokens": 2}}}},
            {"timestamp": "2026-01-01T00:00:07Z", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done"}]}},
            {"timestamp": "2026-01-01T00:00:08Z", "type": "event_msg", "payload": {"type": "task_complete", "turn_id": "turn-1"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            bundle = convert_rollouts([path], {"turn-1"})
        self.assertEqual(len(bundle["traces"]), 1)
        self.assertEqual(bundle["traces"][0]["status"], "success")
        self.assertEqual(bundle["traces"][0]["usage"]["input"], 10)
        self.assertEqual(len(bundle["tool_calls"]), 1)
        self.assertEqual(len(bundle["generations"]), 1)
        roots = [row for row in bundle["observations"] if row["parent_observation_id"] is None]
        self.assertEqual(len(roots), 1)

    def test_subagent_rollout_is_rejected(self) -> None:
        row = {"timestamp": "2026-01-01T00:00:00Z", "type": "session_meta", "payload": {"id": "sub", "source": {"subagent": {"other": "guardian"}}, "thread_source": "subagent"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                convert_rollouts([path])


if __name__ == "__main__":
    unittest.main()
