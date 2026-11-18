"""Export a completed AgRoBERTa run into Phase 1 artifacts. No neural execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from _common import PHASE
from agxqa.config import ADAPTER, ADAPTER_REVISION, BACKBONE, BACKBONE_REVISION, DATA_REVISION, DATASET, PAPER
from agxqa.data import load_data
from agxqa.io import sha256, write_json
from agxqa.metrics import score


def read(path):
    return json.loads(Path(path).read_text())


def export(source, data_dir, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    splits, audit = load_data(data_dir)
    training = read(source / "training/training_metadata.json")
    history = read(source / "training/history.json")
    smoke = read(source / "smoke/smoke-report.json")
    if training.get("completed_epochs") != 20 or training.get("completed_steps") != 7900:
        raise ValueError("Expected the completed 20-epoch / 7900-step reproduction")
    if len(history) != 20 or smoke["status"] != "passed":
        raise ValueError("Incomplete epoch history or failed smoke check")
    if not training["full_20_epoch_published_run"]:
        raise ValueError("Run did not use the recorded default training configuration")
    source_hashes = read(source / "execution.json")["source_sha256"]
    for name, expected in source_hashes.items():
        if sha256(PHASE / "src" / name) != expected:
            raise ValueError(f"Vendored training source differs from the executed source: {name}")
    for split, report in audit["splits"].items():
        report["path"] = "data/agxqa_v1/" + Path(report["path"]).name
    write_json(output / "dataset_statistics.json", {"dataset": DATASET, "revision": DATA_REVISION,
               "language": "en", "task": "extractive_qa", "domain": "agriculture_irrigation",
               "total_qa_pairs": sum(len(rows) for rows in splits.values()), "audit": audit})
    write_json(provenance / "test_ids.json", {"dataset_revision": DATA_REVISION, "ids": [r["id"] for r in splits["test"]]})
    settings = {"model_name": BACKBONE, "task": "extractive_qa", "experiment_status": "completed",
                "dataset": DATASET, "dataset_revision": DATA_REVISION, "data_dir": "data/agxqa_v1",
                "model": {"family": "agroberta_adapter", "backbone": BACKBONE, "backbone_revision": BACKBONE_REVISION,
                          "adapter": ADAPTER, "adapter_revision": ADAPTER_REVISION},
                "training": training["training"],
                "preprocessing": {"max_seq_length": training["decode"]["max_length"], "doc_stride": training["decode"]["stride"],
                                  "padding": "max_length", "question_context_order": "question_then_context"},
                "evaluation": {"protocol": "squad_v1_english", "metrics": ["exact_match", "f1"],
                               "n_best": training["decode"]["n_best"], "max_answer_length": training["decode"]["max_answer_length"],
                               "allow_no_answer": training["decode"]["allow_no_answer"]}}
    import yaml
    (PHASE / "configs/baseline_config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    proposed = {"experiment_status": "proposed_not_run", "model_name": "xlm-roberta-base", "task": "extractive_qa",
                "dataset": DATASET, "dataset_revision": DATA_REVISION,
                "training": {"epochs": 5, "learning_rate": 0.00003, "batch_size": 16, "max_seq_length": 384,
                             "doc_stride": 128, "weight_decay": 0.01, "seed": 42},
                "evaluation": {"metrics": ["exact_match", "f1"]},
                "notes": "User-suggested starting configuration, not original-paper settings or a completed experiment. Model revision must be pinned before execution."}
    (PHASE / "configs/xlmr_proposed_config.yaml").write_text(yaml.safe_dump(proposed, sort_keys=False))
    with (output / "training_logs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "optimizer_step", "training_loss", "validation_loss", "validation_exact_match", "validation_f1"], lineterminator="\n")
        writer.writeheader()
        for row in history:
            writer.writerow({"epoch": int(row["epoch"]), "optimizer_step": row["global_step"],
                             "training_loss": row["mean_training_loss"], "validation_loss": "",
                             "validation_exact_match": row["validation"]["exact_match"], "validation_f1": row["validation"]["f1"]})
    aggregate, native_details, predictions_by_stage = {}, {}, {}
    for folder, filename, model in (("final-test", "test_predictions.jsonl", "agroberta_reimplementation"),
                                    ("baseline-test", "before_finetuning_predictions.jsonl", "agroberta_squad_initialization")):
        predictions = read(source / folder / "predictions.json")
        metrics, details = score(splits["test"], predictions)
        stored = read(source / folder / "metrics.json")
        for name in ("exact_match", "f1", "total"):
            if abs(metrics[name] - stored[name]) > 1e-9:
                raise ValueError(f"Saved metrics differ from rescoring: {folder}/{name}")
        detail_map = {d["id"]: d for d in details}
        with (output / filename).open("w") as stream:
            for gold in splits["test"]:
                d = detail_map[gold["id"]]
                row = {"id": gold["id"], "language": "en", "model": model, "question": gold["question"],
                       "gold_answer": gold["answers"]["text"][0], "gold_answers": gold["answers"]["text"],
                       "prediction_text": d["prediction"], "predicted_answer": d["prediction"],
                       "exact_match": d["exact_match"], "f1": d["f1"], "evaluation_protocol": "squad_v1_english",
                       "dataset_revision": DATA_REVISION, "source_run_id": source.name}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        aggregate[folder] = metrics
        native_details[folder] = details
        predictions_by_stage[folder] = predictions
    correct = sum(d["exact_match"] == 1 for d in native_details["final-test"])
    partial = sum(d["exact_match"] != 1 and d["f1"] > 0 for d in native_details["final-test"])
    no_overlap = sum(d["exact_match"] != 1 and d["f1"] == 0 for d in native_details["final-test"])
    write_json(output / "error_analysis.json", {"model": "agroberta_reimplementation", "protocol": "squad_v1_english",
               "test_questions": 330, "normalized_exact_answers": correct, "nonexact_with_token_overlap": partial,
               "nonexact_with_zero_token_overlap": no_overlap,
               "lowest_f1_nonexact_ids": [d["id"] for d in sorted(native_details["final-test"], key=lambda d: d["f1"]) if d["exact_match"] != 1][:15],
               "limits": "Descriptive analysis of frozen predictions; no test-set tuning or removal of bad gold offsets."})
    parameter_counts = {"total": sum(training["trainability"][k] for k in ("trainable_parameters", "frozen_parameters")),
                        "trainable": training["trainability"]["trainable_parameters"], "frozen": training["trainability"]["frozen_parameters"]}
    summary = {"phase": 1, "phase_status": "partial", "owner": "Saunak", "reviewer": "Dhiraj",
               "dataset": DATASET, "dataset_revision": DATA_REVISION, "language": "en", "test_count": 330,
               "evaluation_protocol": "squad_v1_english", "published_reference": PAPER,
               "before_finetuning": aggregate["baseline-test"], "agroberta_reimplementation": aggregate["final-test"],
               "difference_from_paper_percentage_points": {k: aggregate["final-test"][k] - PAPER[k] for k in PAPER},
               "training": {"epochs": 20, "optimizer_steps": 7900, "selection_rule": training["selection_rule"],
                            "training_time_seconds_monotonic": training["elapsed_seconds"], "device": training["model"]["device"],
                            "parameter_counts": parameter_counts, "validation_loss_recorded": False},
               "authors_released_checkpoint_evaluation": {"status": "unavailable_in_supplied_audit", "original_trained_qa_weights_evaluated": False},
               "xlmr_english_training": {"status": "not_run", "checkpoint_url": None},
               "checkpoint_publication": {"agroberta_status": "local_only_not_uploaded", "agroberta_url": None},
               "claim_limit": "One fixed-seed reimplementation; 309/330 test contexts occur verbatim in train; original decoding settings unavailable."}
    write_json(output / "baseline_results.json", summary)
    with (output / "model_comparison.csv").open("w", newline="") as stream:
        fields = ["model", "status", "evaluation_protocol", "exact_match", "f1", "training_time_seconds", "total_parameters", "trainable_parameters", "notes"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({"model": "AgRoBERTa (published)", "status": "published_reference", "evaluation_protocol": "paper_reported", **PAPER,
                         "notes": "Published values; not measured by this project"})
        writer.writerow({"model": "AgRoBERTa + SQuAD initialization", "status": "measured_before_finetuning", "evaluation_protocol": "squad_v1_english",
                         "exact_match": aggregate["baseline-test"]["exact_match"], "f1": aggregate["baseline-test"]["f1"],
                         "notes": "Agricultural backbone plus existing public SQuAD adapter/head"})
        writer.writerow({"model": "AgRoBERTa (reimplementation)", "status": "measured", "evaluation_protocol": "squad_v1_english",
                         "exact_match": aggregate["final-test"]["exact_match"], "f1": aggregate["final-test"]["f1"],
                         "training_time_seconds": training["elapsed_seconds"], "total_parameters": parameter_counts["total"],
                         "trainable_parameters": parameter_counts["trainable"], "notes": "20 epochs; final checkpoint; not original authors weights"})
        for model in ("XLM-R Base", "RoBERTa Base", "BERT Base", "DistilBERT"):
            writer.writerow({"model": model, "status": "not_run", "notes": "No measured scores or checkpoint in this contribution"})
    write_json(provenance / "run_metadata.json", {"source_run_id": source.name, "training": training["training"], "decode": training["decode"],
               "model": settings["model"], "environment": training["environment"], "parameter_counts": parameter_counts,
               "feature_counts": {"train": training["features"]["features"],
                                  "validation": read(source/"training/validation-epoch-20/run.json")["features"]["features"],
                                  "test": read(source/"final-test/run.json")["features"]["features"]},
               "boundary_whitespace_training_rows": len(training["features"]["answer_boundary_whitespace_trimmed_for_labels"]),
               "frozen_parameter_verification": training["verification"], "smoke_verification": {k:v for k,v in smoke.items() if k != "checkpoint"},
               "recorded_core_source_sha256": {str(Path("src") / k): v for k, v in source_hashes.items()},
               "original_source_artifact_sha256": {name:sha256(source/name) for name in ("training/training_metadata.json", "training/history.json", "final-test/predictions.json", "final-test/metrics.json", "baseline-test/predictions.json")},
               "privacy_note": "Original personal workspace paths and execution authorization messages were omitted; measured settings/results are unchanged."})
    checkpoint = source / "training/final"
    write_json(provenance / "checkpoint_manifest.json", {"status": "local_only_not_uploaded", "public_url": None,
               "source_run_id": source.name, "relative_checkpoint": "training/final", "requires_backbone": BACKBONE,
               "backbone_revision": BACKBONE_REVISION,
               "files": {p.name: {"bytes":p.stat().st_size,"sha256":sha256(p)} for p in checkpoint.iterdir() if p.is_file()},
               "weights_in_git": False})
    # Optional metric-only bridge to the repository's distinct multilingual protocol.
    sys.path.insert(0, str(PHASE.parent))
    from shared.metrics import best_scores
    shared_report = {"protocol": "agxqa-indic-v1", "method": "Metric-only rescore of frozen predictions using shared.metrics.best_scores; no gold-offset repair or model execution", "paper_comparable": False, "models": {}}
    for folder, predictions in predictions_by_stage.items():
        values = [best_scores(predictions[r["id"]], r["answers"]["text"]) for r in splits["test"]]
        shared_report["models"][folder] = {"count": len(values), "exact_match":100*sum(v["em"] for v in values)/len(values),
                                           "f1":100*sum(v["f1"] for v in values)/len(values)}
    write_json(output / "shared_protocol_results.json", shared_report)
    checks = [("Original paper and dataset references", "complete"), ("Dataset download and preprocessing code", "complete"),
              ("Dataset statistics", "complete"), ("AgRoBERTa checkpoint evaluation", "partial"),
              ("English XLM-R fine-tuning", "pending"), ("Training and evaluation scripts", "partial"),
              ("Actual hyperparameters and random seeds", "complete"), ("EM and F1 results for mandatory models", "partial"),
              ("Published vs reproduced comparison", "complete"), ("Question-wise test predictions", "complete"),
              ("Training logs", "complete"), ("Training and evaluation graphs", "partial"),
              ("Fine-tuned XLM-R checkpoint link", "pending"), ("Complete README", "complete")]
    write_json(output / "completion_tracker.json", {"phase_status":"partial", "complete":sum(v == "complete" for _,v in checks),
               "partial":sum(v == "partial" for _,v in checks), "pending":sum(v == "pending" for _,v in checks),
               "total":14, "items":[{"item":k,"status":v} for k,v in checks],
               "notes": ["Original authors QA weights unavailable; our reimplementation checkpoint was evaluated.",
                         "Scripts support the measured AgRoBERTa run; XLM-R training is not implemented/run.",
                         "Validation loss was not logged; its PNG is an availability notice, not a measured curve."]})
    print(json.dumps({"output":str(output), "predictions":330, "scores":aggregate["final-test"], "errors":{ "exact":correct, "partial":partial, "zero_overlap":no_overlap}},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PHASE/"results")
    args=parser.parse_args()
    export(args.source_run,args.data_dir,args.output_dir)
