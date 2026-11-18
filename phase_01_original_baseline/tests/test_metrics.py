import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest

from agxqa.metrics import exact_match, normalize, score, token_f1


class MetricsTests(unittest.TestCase):
    def test_squad_normalization_and_repeated_tokens(self):
        self.assertEqual(normalize(" The, GREEN   plants! "), "green plants")
        self.assertEqual(exact_match("The wheat.", "wheat"), 1.0)
        self.assertAlmostEqual(token_f1("soil soil crop", "soil crop"), 0.8)
        self.assertEqual(token_f1("water", "soil"), 0.0)

    def test_known_example_and_best_reference(self):
        rows = [
            {"id": "1", "answers": {"text": ["green plants", "healthy wheat"]}},
            {"id": "2", "answers": {"text": ["red clay soil"]}},
        ]
        metrics, details = score(rows, {"1": "The healthy wheat!", "2": "clay soil"})
        self.assertEqual(metrics["exact_match"], 50.0)
        self.assertAlmostEqual(metrics["f1"], 90.0)
        self.assertEqual(details[0]["f1"], 1.0)

    def test_coverage_rejects_missing_and_extra_ids(self):
        rows = [{"id": "1", "answers": {"text": ["soil"]}}]
        for predictions in ({}, {"1": "soil", "extra": "soil"}):
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                score(rows, predictions)

    def test_gold_offset_is_not_used_for_scoring(self):
        rows = [{"id": "1", "answers": {"text": ["soil"], "answer_start": [999]}}]
        self.assertEqual(score(rows, {"1": "soil"})[0]["exact_match"], 100.0)

    def test_empty_split_and_duplicate_gold_ids_fail(self):
        with self.assertRaises(ValueError):
            score([], {})
        row = {"id": "1", "answers": {"text": ["soil"]}}
        with self.assertRaises(ValueError):
            score([row, row], {"1": "soil"})


if __name__ == "__main__":
    unittest.main()
