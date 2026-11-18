"""Download and verify the pinned published English splits; no model execution."""
from _common import arguments, data_path, read_config
from agxqa.config import DATA_REVISION, DATASET
from agxqa.data import download_data


def main():
    args = arguments(__doc__).parse_args()
    config = read_config(args.config)
    if config["dataset"] != DATASET or config["dataset_revision"] != DATA_REVISION:
        raise ValueError("Configuration must use the pinned official dataset revision.")
    report = download_data(data_path(args, config))
    print({split: details["rows"] for split, details in report["splits"].items()})


if __name__ == "__main__":
    main()
