import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

"""Synthetic tokenization and logits only: no model, downloads, or inference."""
import unittest

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from agxqa.config import DecodeSettings
from agxqa.features import decode_spans, prepare_features


def toy_tokenizer():
    backend = Tokenizer(WordLevel({"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
                                  "what": 4, "soil": 5, "harvest": 6}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B:1 </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", cls_token="<s>",
                                  eos_token="</s>", sep_token="</s>", pad_token="<pad>", unk_token="<unk>",
                                  model_input_names=["input_ids", "attention_mask"])


def row():
    context = "soil " * 25 + "harvest"
    return {"id": "long", "question": "what", "context": context,
            "answers": {"text": ["harvest"], "answer_start": [context.index("harvest")]}}


class FeaturesTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = toy_tokenizer()
        self.settings = DecodeSettings(max_length=16, stride=4)

    def test_overflow_labels_original_offsets_and_merge(self):
        example = row()
        features, report = prepare_features([example], self.tokenizer, self.settings, training=True)
        self.assertGreater(len(features), 1)
        self.assertEqual(report["used_rows"], 1)
        self.assertEqual(features[0]["inputs"]["start_positions"], features[0]["cls_index"])
        starts, ends = [], []
        for feature in features:
            self.assertIsNone(feature["offsets"][1])  # Question token.
            start, end = np.full(16, -10.0), np.full(16, -10.0)
            # Question/special-token logits must never become span candidates.
            start[0:2] = end[0:2] = 100.0
            for index, offset in enumerate(feature["offsets"]):
                if offset and example["context"][offset[0]:offset[1]] == "harvest":
                    start[index] = end[index] = 20.0
                    self.assertEqual(feature["inputs"]["start_positions"], index)
                    self.assertEqual(feature["inputs"]["end_positions"], index)
            starts.append(start)
            ends.append(end)
        predictions, spans = decode_spans([example], features, starts, ends, self.settings)
        self.assertEqual(predictions, {"long": "harvest"})
        self.assertEqual(spans["long"]["start"], example["answers"]["answer_start"][0])

    def test_invalid_training_offsets_are_explicitly_skipped_or_rejected(self):
        good = row()
        bad = {**good, "id": "bad", "answers": {"text": ["harvest"], "answer_start": [0]}}
        features, report = prepare_features([good, bad], self.tokenizer, self.settings, training=True)
        self.assertEqual(report["excluded_invalid_answer_ids"], ["bad"])
        self.assertEqual({f["id"] for f in features}, {"long"})
        with self.assertRaisesRegex(ValueError, "no valid training answer"):
            prepare_features([bad], self.tokenizer, self.settings, training=True, invalid_policy="error")

    def test_stride_must_fit_actual_context_capacity(self):
        with self.assertRaisesRegex(ValueError, "context capacity"):
            prepare_features([row()], self.tokenizer, DecodeSettings(max_length=16, stride=12))

    def test_span_constraints_and_optional_null_threshold(self):
        example = {"id": "short", "question": "what", "context": "soil harvest",
                   "answers": {"text": ["harvest"], "answer_start": [5]}}
        features, _ = prepare_features([example], self.tokenizer, self.settings)
        feature = features[0]
        context_indices = [i for i, v in enumerate(feature["offsets"]) if v]
        start, end = np.full(16, -20.0), np.full(16, -20.0)
        start[context_indices[0]], end[context_indices[1]] = 10.0, 10.0
        start[context_indices[1]], end[context_indices[0]] = 2.0, 1.0
        start[0] = end[0] = 50.0
        settings = DecodeSettings(max_length=16, stride=4, max_answer_length=1)
        prediction, _ = decode_spans([example], features, [start], [end], settings)
        self.assertEqual(prediction["short"], "harvest")  # Disallows the higher-score two-token span.
        settings.allow_no_answer = True
        prediction, _ = decode_spans([example], features, [start], [end], settings)
        self.assertEqual(prediction["short"], "")
        start[0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            decode_spans([example], features, [start], [end], settings)

    def test_answer_too_long_to_fit_any_window_fails(self):
        example = row()
        example["answers"] = {"text": [example["context"]], "answer_start": [0]}
        with self.assertRaisesRegex(ValueError, "fit no window"):
            prepare_features([example], self.tokenizer, self.settings, training=True)

    def test_zero_length_offsets_inside_context_are_skipped_for_labels(self):
        class TokenizerWithEmptyOffset:
            def __init__(self, base):
                self.base = base

            def __getattr__(self, name):
                return getattr(self.base, name)

            def __call__(self, *args, **kwargs):
                encoded = self.base(*args, **kwargs)
                if kwargs.get("return_offsets_mapping"):
                    for i, offsets in enumerate(encoded["offset_mapping"]):
                        positions = [j for j, seq in enumerate(encoded.sequence_ids(i)) if seq == 1]
                        hole = positions[1]
                        offsets[hole] = (offsets[hole][0], offsets[hole][0])
                return encoded

        example = {"id": "holes", "question": "what", "context": "soil soil harvest",
                   "answers": {"text": ["harvest"], "answer_start": [10]}}
        features, _ = prepare_features([example], TokenizerWithEmptyOffset(self.tokenizer), self.settings, training=True)
        feature = features[0]
        self.assertIsNone(feature["offsets"][5])
        self.assertEqual(feature["offsets"][feature["inputs"]["start_positions"]], (10, 17))
        self.assertEqual(feature["inputs"]["start_positions"], feature["inputs"]["end_positions"])

    def test_answer_boundary_whitespace_is_ignored_only_for_token_labels(self):
        example = {"id": "whitespace", "question": "what", "context": "soil harvest ",
                   "answers": {"text": [" harvest "], "answer_start": [4]}}
        features, report = prepare_features([example], self.tokenizer, self.settings, training=True)
        feature = features[0]
        self.assertEqual(feature["offsets"][feature["inputs"]["start_positions"]], (5, 12))
        self.assertEqual(feature["inputs"]["start_positions"], feature["inputs"]["end_positions"])
        self.assertEqual(report["answer_boundary_whitespace_trimmed_for_labels"], ["whitespace"])
        self.assertEqual(example["answers"], {"text": [" harvest "], "answer_start": [4]})


if __name__ == "__main__":
    unittest.main()
