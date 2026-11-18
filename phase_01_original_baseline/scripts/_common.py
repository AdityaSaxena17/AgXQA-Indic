"""Phase-local configuration; importing this file does not execute a model."""
import argparse
from pathlib import Path
import sys

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))


def arguments(description, model=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=PHASE / "configs/baseline_config.yaml")
    parser.add_argument("--data-dir", type=Path)
    if model:
        parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--base-model")
        parser.add_argument("--adapter")
        parser.add_argument("--tokenizer")
        parser.add_argument("--head")
    return parser


def read_config(path):
    import yaml
    config = yaml.safe_load(path.read_text())
    if config.get("model", {}).get("family") != "agroberta_adapter":
        raise ValueError("These scripts reproduce AgRoBERTa adapters. XLM-R remains a proposed, unrun experiment.")
    if config.get("evaluation", {}).get("protocol") != "squad_v1_english":
        raise ValueError("Use the recorded squad_v1_english protocol for this paper comparison.")
    from agxqa.config import DATASET, DATA_REVISION
    if config.get("dataset") != DATASET or config.get("dataset_revision") != DATA_REVISION:
        raise ValueError("Use the pinned official AgXQA dataset and revision.")
    fixed = {"optim": "adamw_torch", "lr_scheduler_type": "linear", "fp16": False, "bf16": False,
             "load_best_model_at_end": False, "max_steps": None, "train_limit": None, "eval_limit": None}
    for key, expected in fixed.items():
        if config["training"].get(key) != expected:
            raise ValueError(f"Unsupported training setting {key}; this reproducer requires {expected!r}")
    return config


def data_path(args, config):
    return args.data_dir or PHASE / config["data_dir"]


def core_options(args, config):
    model, prep, evaluation = config["model"], config["preprocessing"], config["evaluation"]
    options = ["--data-dir", str(data_path(args, config)),
               "--base-model", args.base_model or model["backbone"],
               "--base-revision", model["backbone_revision"],
               "--adapter", args.adapter or model["adapter"],
               "--adapter-revision", model["adapter_revision"],
               "--device", args.device, "--output-dir", str(args.output_dir),
               "--max-length", str(prep["max_seq_length"]), "--stride", str(prep["doc_stride"]),
               "--n-best", str(evaluation["n_best"]),
               "--max-answer-length", str(evaluation["max_answer_length"]),
               "--seed", str(config["training"]["seed"]),
               "--eval-batch-size", str(config["training"]["eval_batch_size"])]
    for key in ("tokenizer", "head"):
        if getattr(args, key):
            options.extend(["--" + key, getattr(args, key)])
    if evaluation.get("allow_no_answer", False):
        options.append("--allow-no-answer")
    return options
