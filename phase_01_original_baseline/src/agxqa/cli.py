import argparse
import gc
import json
from dataclasses import fields
from pathlib import Path

from .config import ADAPTER, ADAPTER_REVISION, BACKBONE, BACKBONE_REVISION, COUNTS, DecodeSettings, TrainSettings
from .data import download_data, load_data, subset
from .io import new_run_dir, write_json
from .metrics import score


def _decode(args):
    return DecodeSettings(**{f.name: getattr(args, f.name) for f in fields(DecodeSettings)})


def _training(args):
    defaults = TrainSettings()
    return TrainSettings(**{f.name: getattr(args, f.name, getattr(defaults, f.name)) for f in fields(TrainSettings)})


def _add_runtime(parser):
    parser.add_argument("--base-model", default=BACKBONE, help="Pinned Hub repository or local backbone directory")
    parser.add_argument("--base-revision", default=BACKBONE_REVISION)
    parser.add_argument("--adapter", default=ADAPTER, help="Hub repository or local adapter+head checkpoint")
    parser.add_argument("--adapter-revision", default=ADAPTER_REVISION)
    parser.add_argument("--head", help="Optional separate local QA head directory")
    parser.add_argument("--tokenizer", help="Optional tokenizer directory or Hub repository")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--n-best", type=int, default=20)
    parser.add_argument("--max-answer-length", type=int, default=30)
    parser.add_argument("--allow-no-answer", action="store_true")
    parser.add_argument("--null-score-diff-threshold", type=float, default=0.0)


def _add_training(parser, smoke=False):
    defaults = TrainSettings(epochs=1, max_steps=2, train_limit=8, eval_limit=4) if smoke else TrainSettings()
    for name in ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps", "warmup_steps", "save_steps", "max_steps", "train_limit", "eval_limit"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon", "max_grad_norm"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--selection", choices=["last", "best-validation-f1"], default="last")
    parser.add_argument("--invalid-train-answers", choices=["skip", "error"], default="skip")


def parser():
    root = argparse.ArgumentParser(description="Pinned AgXQA adapter fine-tuning and extractive QA evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download-data")
    fetch.add_argument("--data-dir", default="data")
    fetch_models = commands.add_parser("download-models")
    fetch_models.add_argument("--models-dir", default="models")
    for command in ("validate", "score", "evaluate", "train", "smoke", "reproduce"):
        sub = commands.add_parser(command)
        sub.add_argument("--data-dir", default="data")
        if command == "validate":
            sub.add_argument("--output", default="reports/data-audit.json")
            continue
        if command == "score":
            sub.add_argument("--predictions", required=True)
            sub.add_argument("--split", choices=list(COUNTS), default="test")
            sub.add_argument("--output", default="reports/score.json")
            continue
        sub.add_argument("--output-dir", required=True)
        _add_runtime(sub)
        if command == "evaluate":
            sub.add_argument("--split", choices=list(COUNTS), default="test")
            sub.add_argument("--limit", type=int)
            sub.add_argument("--eval-batch-size", type=int, default=4)
            sub.add_argument("--label", help="Result label; inferred from adapter source if omitted")
        else:
            _add_training(sub, smoke=command == "smoke")
    return root


def _load_kwargs(args, device):
    return {"base": args.base_model, "base_revision": args.base_revision, "adapter": args.adapter,
            "adapter_revision": args.adapter_revision, "head": args.head, "tokenizer_path": args.tokenizer,
            "tokenizer_revision": args.tokenizer_revision, "device": device}


def _checkpoint_training(path):
    record = Path(path) / "training_metadata.json"
    return json.loads(record.read_text()) if record.exists() else None


def execute(args):
    if args.command == "download-data":
        report = download_data(args.data_dir)
        print(json.dumps({k: v["rows"] for k, v in report["splits"].items()}))
        return
    if args.command == "download-models":
        from .model import download_models
        print(json.dumps(download_models(args.models_dir), indent=2))
        return
    splits, data_report = load_data(args.data_dir)
    if args.command == "validate":
        write_json(args.output, data_report)
        print(json.dumps({k: {"rows": v["rows"], "invalid_offsets": len(v["invalid_offsets"])}
                          for k, v in data_report["splits"].items()}))
        return
    if args.command == "score":
        def no_duplicate_keys(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"Duplicate prediction ID: {key}")
                value[key] = item
            return value
        predictions = json.loads(Path(args.predictions).read_text(), object_pairs_hook=no_duplicate_keys)
        metrics, _ = score(splits[args.split], predictions)
        write_json(args.output, metrics)
        print(json.dumps(metrics))
        return
    from .model import configure_runtime, load_model
    from .workflow import evaluate_model, run_smoke, train_model
    device = configure_runtime(args.device, args.threads, args.seed)
    decode = _decode(args)
    decode.validate()
    kwargs = _load_kwargs(args, device)
    if args.command == "smoke":
        settings = _training(args)
        if settings.selection != "last":
            raise ValueError("Smoke verification requires --selection last")
        report = run_smoke(kwargs, splits, decode, settings, device, args.output_dir, data_report)
        print(json.dumps(report, indent=2))
        return
    if args.command == "evaluate":
        new_run_dir(args.output_dir)
        model, tokenizer, metadata = load_model(**kwargs)
        label = args.label or ("before fine-tuning" if args.adapter == ADAPTER or Path(args.adapter).name == "squad"
                               else "AgRoBERTa reimplementation")
        evaluate_model(model, tokenizer, subset(splits[args.split], args.limit), decode,
                       args.eval_batch_size, device, args.output_dir, label, args.split, metadata, data_report,
                       training=_checkpoint_training(args.adapter))
        return
    settings = _training(args)
    settings.validate()
    output = new_run_dir(args.output_dir)
    if args.command == "reproduce":
        # Small integration check precedes any full baseline or training run.
        smoke_settings = TrainSettings(epochs=1, max_steps=2, train_limit=8, eval_limit=4, seed=args.seed)
        run_smoke(kwargs, splits, decode, smoke_settings, device, output / "smoke", data_report)
        gc.collect()
        configure_runtime(args.device, args.threads, args.seed)
    model, tokenizer, metadata = load_model(**kwargs)
    if args.command == "reproduce":
        baseline = evaluate_model(model, tokenizer, splits["test"], decode, settings.eval_batch_size, device,
                                  output / "baseline-test", "before fine-tuning", "test", metadata, data_report)
        train_output = output / "training"
    else:
        train_output = output
    trained = train_model(model, tokenizer, splits, decode, settings, device, train_output, metadata, data_report)
    if args.command == "reproduce":
        model.to("cpu")
        del model
        gc.collect()
        kwargs.update({"adapter": trained["checkpoint"], "head": None, "tokenizer_path": trained["checkpoint"]})
        model, tokenizer, metadata = load_model(**kwargs)
        final = evaluate_model(model, tokenizer, splits["test"], decode, settings.eval_batch_size, device,
                               output / "final-test", "AgRoBERTa reimplementation", "test", metadata,
                               data_report, training=trained["record"])
        write_json(output / "comparison.json", {"paper_published_reference": {"exact_match": 55.15, "f1": 78.89},
                   "before_fine_tuning": baseline["metrics"], "agroberta_reimplementation": final["metrics"],
                   "full_20_epoch_published_run": trained["record"]["full_20_epoch_published_run"]})
    print(json.dumps({"selected_checkpoint": trained["checkpoint"], "verification": trained["record"]["verification"]}))


def main():
    root = parser()
    args = root.parse_args()
    try:
        execute(args)
    except (ValueError, FileNotFoundError) as error:
        root.exit(2, f"agxqa: {error}\n")
