import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest

from agxqa.cli import _decode, _training, parser
from agxqa.config import TrainSettings


class ConfigTests(unittest.TestCase):
    def test_published_defaults(self):
        args = parser().parse_args(["train", "--output-dir", "unused"])
        self.assertEqual(_training(args), TrainSettings())
        decode = _decode(args)
        self.assertEqual((decode.max_length, decode.stride, decode.n_best, decode.max_answer_length), (384, 128, 20, 30))
        self.assertFalse(decode.allow_no_answer)

    def test_smoke_limits_and_overrides(self):
        args = parser().parse_args(["smoke", "--output-dir", "unused", "--max-length", "256", "--max-steps", "3"])
        self.assertEqual(_training(args).max_steps, 3)
        self.assertEqual(_training(args).train_limit, 8)
        self.assertEqual(_training(args).eval_limit, 4)
        self.assertEqual(_decode(args).max_length, 256)

    def test_invalid_training_limits(self):
        with self.assertRaises(ValueError):
            TrainSettings(max_steps=0).validate()


if __name__ == "__main__":
    unittest.main()
