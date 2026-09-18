import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIMULATION_DIR = ROOT / "simulation"


class SyntheticGeneratorTests(unittest.TestCase):
    def generate(self, output: Path) -> None:
        subprocess.run([
            sys.executable,
            str(SIMULATION_DIR / "generate_synthetic_corpus.py"),
            "--output-dir", str(output),
            "--task-count", "24",
            "--model-count", "2",
            "--seed", "17",
            "--bootstrap-repetitions", "20",
            "--permutation-repetitions", "20",
        ], check=True, capture_output=True, text=True)

    def test_generation_is_reproducible_and_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            self.generate(first)
            self.generate(second)
            for filename in ("traces.jsonl", "observations.jsonl", "model_answers.jsonl", "annotations.jsonl"):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            completed = subprocess.run([
                sys.executable,
                str(SIMULATION_DIR / "validate_synthetic_corpus.py"),
                "--data-dir", str(first),
            ], check=True, capture_output=True, text=True)
            self.assertTrue(json.loads(completed.stdout)["passed"])

    def test_private_truth_declares_synthetic_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "data"
            self.generate(output)
            truth = json.loads((output / "private_ground_truth.json").read_text())
            self.assertEqual(truth["generator_version"], "synthetic-trace-generator-v1")
            self.assertGreater(truth["counts"]["traces"], 0)
            self.assertEqual(len(truth["models"]), 2)


if __name__ == "__main__":
    unittest.main()
