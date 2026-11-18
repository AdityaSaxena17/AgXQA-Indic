"""Train the AgRoBERTa adapter/head using the recorded configuration. Executes a model."""
from _common import arguments, core_options, read_config


def main():
    parser = arguments(__doc__, model=True)
    parser.add_argument("--smoke", action="store_true", help="Two updates on eight train/four validation rows")
    parser.add_argument("--reproduce", action="store_true", help="Smoke, baseline test, full train, final test")
    args = parser.parse_args()
    if args.smoke and args.reproduce:
        parser.error("Choose at most one of --smoke and --reproduce")
    config = read_config(args.config)
    command = "smoke" if args.smoke else "reproduce" if args.reproduce else "train"
    options = [command, *core_options(args, config)]
    settings = config["training"]
    for key in ("learning_rate", "batch_size", "gradient_accumulation_steps", "weight_decay", "adam_beta1", "adam_beta2",
                "adam_epsilon", "max_grad_norm", "warmup_steps", "save_steps", "selection", "invalid_train_answers"):
        options.extend(["--" + key.replace("_", "-"), str(settings[key])])
    if not args.smoke:
        options.extend(["--epochs", str(settings["epochs"])])
    from agxqa.cli import execute, parser as core_parser
    execute(core_parser().parse_args(options))


if __name__ == "__main__":
    main()
