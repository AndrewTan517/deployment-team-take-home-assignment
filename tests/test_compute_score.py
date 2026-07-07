import json
import tempfile
import unittest

from src.compute_score import (
    compute_score,
    exact_match_score,
    f1_score,
    normalize_answer,
)


class TestNormalizeAnswer(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(normalize_answer("Hello World"), "hello world")

    def test_removes_punctuation(self):
        self.assertEqual(normalize_answer("Hello, World!"), "hello world")

    def test_removes_articles_a(self):
        self.assertEqual(normalize_answer("a cat"), "cat")

    def test_removes_articles_an(self):
        self.assertEqual(normalize_answer("an apple"), "apple")

    def test_removes_articles_the(self):
        self.assertEqual(normalize_answer("the answer"), "answer")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_answer("  foo   bar  "), "foo bar")

    def test_combined(self):
        self.assertEqual(normalize_answer("The Quick, Brown Fox!"), "quick brown fox")


class TestF1Score(unittest.TestCase):
    def test_exact_match_returns_one(self):
        self.assertAlmostEqual(f1_score("New York City", "New York City"), 1.0)

    def test_no_overlap_returns_zero(self):
        self.assertEqual(f1_score("Paris", "London"), 0)

    def test_partial_overlap(self):
        score = f1_score("New York", "New York City")
        self.assertGreater(score, 0)
        self.assertLess(score, 1)

    def test_normalized_match(self):
        # "the" is an article stripped by normalize_answer
        self.assertAlmostEqual(f1_score("the cat", "cat"), 1.0)

    def test_empty_prediction_returns_zero(self):
        self.assertEqual(f1_score("", "some answer"), 0)


class TestExactMatchScore(unittest.TestCase):
    def test_identical_strings(self):
        self.assertTrue(exact_match_score("Paris", "Paris"))

    def test_case_insensitive(self):
        self.assertTrue(exact_match_score("paris", "Paris"))

    def test_punctuation_ignored(self):
        self.assertTrue(exact_match_score("Paris!", "Paris"))

    def test_article_ignored(self):
        self.assertTrue(exact_match_score("The Eiffel Tower", "Eiffel Tower"))

    def test_different_answers(self):
        self.assertFalse(exact_match_score("London", "Paris"))


class TestComputeScore(unittest.TestCase):
    def _write_dataset(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        f.close()
        return f.name

    def test_exact_match_gets_full_score(self):
        path = self._write_dataset([{"id": "q1", "expected": "Paris"}])
        scores = compute_score(path, {"q1": "Paris"})
        self.assertTrue(scores["q1"]["exact_match"])
        self.assertAlmostEqual(scores["q1"]["f1"], 1.0)

    def test_wrong_answer_gets_zero(self):
        path = self._write_dataset([{"id": "q1", "expected": "Paris"}])
        scores = compute_score(path, {"q1": "London"})
        self.assertFalse(scores["q1"]["exact_match"])
        self.assertEqual(scores["q1"]["f1"], 0)

    def test_unanswered_question_omitted_from_scores(self):
        path = self._write_dataset([{"id": "q1", "expected": "Paris"}])
        scores = compute_score(path, {})
        self.assertNotIn("q1", scores)

    def test_multiple_questions(self):
        path = self._write_dataset([
            {"id": "q1", "expected": "Paris"},
            {"id": "q2", "expected": "London"},
        ])
        scores = compute_score(path, {"q1": "Paris", "q2": "London"})
        self.assertEqual(len(scores), 2)
        self.assertTrue(scores["q1"]["exact_match"])
        self.assertTrue(scores["q2"]["exact_match"])


if __name__ == "__main__":
    unittest.main()
