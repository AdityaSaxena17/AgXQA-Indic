"""Rescore frozen JSONL predictions, or explicitly run checkpoint inference."""
import json
from pathlib import Path
from _common import arguments, core_options, data_path, read_config
from agxqa.data import load_data
from agxqa.io import write_json
from agxqa.metrics import score


def main():
    parser = arguments(__doc__)
    parser.add_argument("--predictions", type=Path, help="Published Phase 1 JSONL; no model loading")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-model", action="store_true", help="Explicitly execute neural checkpoint inference")
    parser.add_argument("--checkpoint")
    parser.add_argument("--base-model")
    parser.add_argument("--adapter")
    parser.add_argument("--tokenizer")
    parser.add_argument("--head")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--label", default="AgRoBERTa reimplementation")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.run_model:
        if args.predictions or not args.output_dir or not args.checkpoint:
            parser.error("--run-model requires --checkpoint and --output-dir; cannot combine with --predictions")
        args.adapter = args.checkpoint or args.adapter
        from agxqa.cli import execute, parser as core_parser
        execute(core_parser().parse_args(["evaluate", *core_options(args, config), "--split", args.split, "--label", args.label]))
        return
    if not args.predictions or not args.output or args.output_dir:
        parser.error("Saved-prediction scoring requires --predictions JSONL and --output JSON")
    splits, _ = load_data(data_path(args, config))
    predictions = {}
    for line in args.predictions.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("language") != "en" or row.get("evaluation_protocol") != "squad_v1_english":
            raise ValueError("Prediction rows must identify English and squad_v1_english")
        if row["id"] in predictions:
            raise ValueError(f"Duplicate prediction ID: {row['id']}")
        predictions[row["id"]] = row["prediction_text"]
    metrics, _ = score(splits[args.split], predictions)
    metrics["evaluation_protocol"] = "squad_v1_english"
    write_json(args.output, metrics)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
