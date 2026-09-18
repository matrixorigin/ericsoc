import unittest
from pathlib import Path

from validation.validate_benchmark_design import validate_catalog


class BenchmarkDesignTests(unittest.TestCase):
    def test_query_design_is_complete_and_consistent(self) -> None:
        summary = validate_catalog()
        self.assertEqual(summary["query_count"], 30)
        self.assertEqual(summary["implemented_count"], 30)
        self.assertEqual(summary["specified_count"], 0)
        self.assertEqual(set(summary["family_counts"].values()), {5})

    def test_each_query_has_one_entry_point(self) -> None:
        root = Path(__file__).resolve().parents[1]
        files = sorted((root / "queries").glob("query[0-9][0-9].py"))
        self.assertEqual([path.stem for path in files], [f"query{number:02d}" for number in range(1, 31)])


if __name__ == "__main__":
    unittest.main()
