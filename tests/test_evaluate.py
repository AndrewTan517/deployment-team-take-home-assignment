import json
import tempfile
import unittest

from src.evaluate import load_dataset, summarize


class TestLoadDataset(unittest.TestCase):
    def test_loads_jsonl(self):
        rows = [{"id": "q1", "input": "What?", "expected": "Yes"}]
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        f.close()
        self.assertEqual(load_dataset(f.name), rows)


class TestSummarize(unittest.TestCase):
    def _write_dataset(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        f.close()
        return f.name

    def test_perfect_run(self):
        path = self._write_dataset([{"id": "q1", "input": "Capital of France?", "expected": "Paris"}])
        result = summarize(path, {"q1": "Paris"}, [])
        self.assertEqual(result["run_metadata"]["total_cases"], 1)
        self.assertEqual(result["run_metadata"]["completed"], 1)
        self.assertEqual(result["run_metadata"]["errored"], 0)
        self.assertAlmostEqual(result["summary"]["pass_rate"], 1.0)
        self.assertAlmostEqual(result["summary"]["mean_f1"], 1.0)
        self.assertEqual(result["failures"], [])

    def test_failed_case_appears_in_failures(self):
        path = self._write_dataset([{"id": "q1", "input": "Capital of France?", "expected": "Paris"}])
        result = summarize(path, {"q1": "London"}, [])
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(result["failures"][0]["id"], "q1")

    def test_error_rate_reflects_errors(self):
        path = self._write_dataset([
            {"id": "q1", "input": "Q1?", "expected": "A"},
            {"id": "q2", "input": "Q2?", "expected": "B"},
        ])
        errors = [{"id": "q2", "input": "Q2?", "error_type": "Timeout", "detail": "timed out"}]
        result = summarize(path, {"q1": "A"}, errors)
        self.assertAlmostEqual(result["summary"]["error_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
