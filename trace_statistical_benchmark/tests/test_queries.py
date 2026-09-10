import unittest
from pathlib import Path

from reference_queries import QUERY_FUNCTIONS, load_data, run_task


DATA_DIR = Path(__file__).resolve().parents[1] / "sample_data"


class QuerySmokeTests(unittest.TestCase):
    def test_all_queries_run_on_sample_data(self) -> None:
        traces, observations = load_data(DATA_DIR)
        self.assertEqual(len(traces), 2)
        self.assertEqual(len(observations), 9)

        for query_id in QUERY_FUNCTIONS:
            with self.subTest(query_id=query_id):
                self.assertIsInstance(run_task(query_id, DATA_DIR), dict)

    def test_status_and_recovery_answers(self) -> None:
        status = run_task("query01", DATA_DIR)
        self.assertEqual(status, {"total_traces": 2, "success": 1, "error": 1, "other": 0})

        recovery = run_task("query11", DATA_DIR)
        self.assertEqual(recovery["failures_with_later_success"], 1)
        self.assertTrue(recovery["rows"][0]["later_success_found"])


if __name__ == "__main__":
    unittest.main()
