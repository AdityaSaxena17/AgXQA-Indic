"""Compatibility entry point; implementation is in scripts/train.py."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    directory = Path(__file__).resolve().parent / "scripts"
    sys.path.insert(0, str(directory))
    runpy.run_path(str(directory / "train.py"), run_name="__main__")
