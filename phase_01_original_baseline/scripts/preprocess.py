"""Audit published files and optionally tokenize windows; no neural model forward pass."""
from _common import PHASE, arguments, data_path, read_config
from agxqa.config import DecodeSettings
from agxqa.data import load_data
from agxqa.io import write_json


def main():
    parser = arguments(__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--tokenizer", help="Local tokenizer directory, otherwise use pinned backbone tokenizer")
    parser.add_argument("--output", type=str, default=str(PHASE / "data/preprocessing_report.json"))
    args = parser.parse_args()
    config = read_config(args.config)
    splits, audit = load_data(data_path(args, config))
    result = {"data": audit, "original_files_modified": False}
    if not args.audit_only:
        from transformers import AutoTokenizer
        from agxqa.features import prepare_features
        model = config["model"]
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model["backbone"], use_fast=True,
                                                  revision=None if args.tokenizer else model["backbone_revision"])
        prep = config["preprocessing"]
        settings = DecodeSettings(max_length=prep["max_seq_length"], stride=prep["doc_stride"])
        result["features"] = {}
        for split, rows in splits.items():
            _, report = prepare_features(rows, tokenizer, settings, training=split == "train")
            result["features"][split] = report
    write_json(args.output, result)
    print({"rows": {k: len(v) for k, v in splits.items()}, "output": args.output})


if __name__ == "__main__":
    main()
