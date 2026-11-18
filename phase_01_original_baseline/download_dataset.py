"""Compatibility entry point; implementation is in scripts/download_dataset.py."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    directory = Path(__file__).resolve().parent / "scripts"
    sys.path.insert(0, str(directory))
    runpy.run_path(str(directory / "download_dataset.py"), run_name="__main__")
