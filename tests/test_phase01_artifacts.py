"""Frozen Phase 1 artifacts: standard-library checks, no model execution/downloads."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "phase_01_original_baseline"
sys.path.insert(0, str(PHASE / "src"))
from agxqa.metrics import score
from shared.metrics import best_scores


def read(name):
    return json.loads((PHASE / "results" / name).read_text())


def predictions(name="test_predictions.jsonl"):
    return [json.loads(line) for line in (PHASE / "results" / name).read_text().splitlines()]


class PhaseOneArtifactTests(unittest.TestCase):
    def test_squad_rescore_matches_measured_aggregate_and_each_row(self):
        summary = read("baseline_results.json")
        for filename, key in (("test_predictions.jsonl", "agroberta_reimplementation"),
                              ("before_finetuning_predictions.jsonl", "before_finetuning")):
            rows = predictions(filename)
            gold = [{"id":r["id"], "answers":{"text":r["gold_answers"]}} for r in rows]
            actual, details = score(gold, {r["id"]:r["prediction_text"] for r in rows})
            self.assertEqual(actual["total"], 330)
            for metric in ("exact_match", "f1"):
                self.assertAlmostEqual(actual[metric], summary[key][metric], places=10)
            for row, detail in zip(rows, details):
                self.assertEqual(row["prediction_text"], row["predicted_answer"])
                self.assertEqual(row["gold_answer"], row["gold_answers"][0])
                self.assertEqual(row["evaluation_protocol"], "squad_v1_english")
                self.assertEqual(row["language"], "en")
                self.assertEqual(row["exact_match"], detail["exact_match"])
                self.assertAlmostEqual(row["f1"], detail["f1"], places=12)

    def test_official_id_coverage_split_counts_and_gold_reference_counts(self):
        rows = predictions()
        ids = [r["id"] for r in rows]
        expected = read("provenance/test_ids.json")["ids"]
        self.assertEqual(len(ids), 330)
        self.assertEqual(len(set(ids)), 330)
        self.assertEqual(set(ids), set(expected))
        self.assertEqual(sum(len(r["gold_answers"]) > 1 for r in rows), 59)
        stats = read("dataset_statistics.json")
        self.assertEqual(stats["total_qa_pairs"], 2186)
        self.assertEqual({k:v["rows"] for k,v in stats["audit"]["splits"].items()}, {"train":1503,"validation":353,"test":330})
        self.assertFalse(stats["audit"]["test_offsets_modified"])
        self.assertEqual(stats["audit"]["train_test_overlap"]["verbatim_context_rows"], 309)

    def test_history_is_complete_and_unrecorded_validation_loss_stays_blank(self):
        with (PHASE / "results/training_logs.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(r["epoch"]) for r in rows], list(range(1,21)))
        self.assertEqual([int(r["optimizer_step"]) for r in rows], [395*n for n in range(1,21)])
        self.assertTrue(all(r["validation_loss"] == "" for r in rows))
        self.assertAlmostEqual(float(rows[-1]["validation_f1"]), 79.8326255827911)

    def test_unrun_models_and_unpublished_checkpoints_are_not_claimed_complete(self):
        with (PHASE / "results/model_comparison.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        unrun = [r for r in rows if r["status"] == "not_run"]
        self.assertEqual(len(unrun), 4)
        self.assertTrue(all(r["exact_match"] == r["f1"] == r["training_time_seconds"] == "" for r in unrun))
        summary = read("baseline_results.json")
        self.assertEqual(summary["phase_status"], "partial")
        self.assertEqual(summary["xlmr_english_training"]["status"], "not_run")
        self.assertIsNone(summary["checkpoint_publication"]["agroberta_url"])
        tracker = read("completion_tracker.json")
        self.assertEqual((tracker["complete"],tracker["partial"],tracker["pending"]), (8,4,2))
        self.assertEqual(len(tracker["items"]), 14)

    def test_saved_runtime_source_matches_executed_hashes(self):
        metadata = read("provenance/run_metadata.json")
        for name, expected in metadata["recorded_core_source_sha256"].items():
            actual = hashlib.sha256((PHASE / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)
        proof = metadata["frozen_parameter_verification"]
        self.assertEqual(proof["frozen_sha256_before"], proof["frozen_sha256_after"])
        self.assertTrue(proof["adapter_updated"] and proof["qa_head_updated"])

    def test_images_exist_and_validation_loss_is_explicitly_unavailable(self):
        for name in ("training_loss", "validation_loss", "baseline_comparison", "f1_across_epochs", "dataset_distribution"):
            raw = (PHASE / "results/plots" / (name + ".png")).read_bytes()
            self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(len(raw), 1000)
        self.assertIn("Availability notice", (PHASE / "results/plots/README.md").read_text())
        self.assertFalse(read("baseline_results.json")["training"]["validation_loss_recorded"])

    def test_secondary_shared_protocol_is_an_independent_labelled_rescore(self):
        report = read("shared_protocol_results.json")
        self.assertFalse(report["paper_comparable"])
        for filename, key in (("test_predictions.jsonl", "final-test"), ("before_finetuning_predictions.jsonl", "baseline-test")):
            rows = predictions(filename)
            values = [best_scores(r["prediction_text"], r["gold_answers"]) for r in rows]
            self.assertAlmostEqual(report["models"][key]["exact_match"], 100*sum(v["em"] for v in values)/330)
            self.assertAlmostEqual(report["models"][key]["f1"], 100*sum(v["f1"] for v in values)/330)


if __name__ == "__main__":
    unittest.main()
