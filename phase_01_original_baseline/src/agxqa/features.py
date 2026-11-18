import math

import numpy as np


def prepare_features(rows, tokenizer, settings, training=False, invalid_policy="skip"):
    settings.validate()
    if not tokenizer.is_fast or tokenizer.padding_side != "right":
        raise ValueError("A fast tokenizer with right padding is required")
    selected, excluded = [], []
    gold = {}
    for row in rows:
        valid = [(start, text) for start, text in zip(row["answers"]["answer_start"], row["answers"]["text"])
                 if start >= 0 and row["context"][start:start + len(text)] == text]
        if training and not valid and row["answers"]["text"]:
            if invalid_policy == "error":
                raise ValueError(f"{row['id']}: no valid training answer offset")
            excluded.append(row["id"])
            continue
        selected.append(row)
        gold[row["id"]] = valid[0] if valid else None
    if not selected:
        raise ValueError("No usable rows remain")
    # Check the actual context capacity, not just the requested total length.
    for row in selected:
        question_length = len(tokenizer(row["question"].lstrip(), add_special_tokens=False)["input_ids"])
        capacity = settings.max_length - question_length - tokenizer.num_special_tokens_to_add(pair=True)
        if capacity <= settings.stride:
            raise ValueError(f"{row['id']}: stride {settings.stride} must be below context capacity {capacity}; increase max_length or reduce stride")
    encoded = tokenizer([r["question"].lstrip() for r in selected], [r["context"] for r in selected],
                        truncation="only_second", max_length=settings.max_length, stride=settings.stride,
                        return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    features, no_positive, trimmed_boundaries = [], set(), set()
    positive = set()
    for i, sample_index in enumerate(encoded["overflow_to_sample_mapping"]):
        row = selected[sample_index]
        sequence_ids = encoded.sequence_ids(i)
        offsets = [tuple(offset) if seq == 1 and offset[1] > offset[0] else None
                   for offset, seq in zip(encoded["offset_mapping"][i], sequence_ids)]
        inputs = {name: encoded[name][i] for name in tokenizer.model_input_names if name in encoded}
        cls_index = inputs["input_ids"].index(tokenizer.cls_token_id)
        feature = {"inputs": inputs, "id": row["id"], "offsets": offsets, "cls_index": cls_index}
        if training:
            start_position = end_position = cls_index
            answer = gold[row["id"]]
            positions = [j for j, offset in enumerate(offsets) if offset is not None]
            if answer and positions:
                start, text = answer
                end = start + len(text)
                # Offsets omit boundary whitespace. Keep the published answer
                # unchanged and align its non-whitespace characters to tokens.
                if text != text.strip():
                    trimmed_boundaries.add(row["id"])
                    start += len(text) - len(text.lstrip())
                    end -= len(text) - len(text.rstrip())
                left, right = positions[0], positions[-1]
                if offsets[left][0] <= start and offsets[right][1] >= end:
                    # Byte-level tokenizers can emit zero-length offsets inside
                    # the context. Walk valid context positions so those holes
                    # cannot become labels or interrupt boundary alignment.
                    start_position = max(p for p in positions if offsets[p][0] <= start)
                    end_position = min(p for p in positions if offsets[p][1] >= end)
                    positive.add(row["id"])
            inputs["start_positions"], inputs["end_positions"] = start_position, end_position
        features.append(feature)
    if training:
        no_positive = {r["id"] for r in selected if gold[r["id"]] is not None} - positive
        if no_positive:
            raise ValueError(f"Training answers fit no window; increase max_length/stride: {sorted(no_positive)}")
    return features, {"input_rows": len(rows), "used_rows": len(selected), "features": len(features),
                      "excluded_invalid_answer_ids": excluded, "answer_choice": "first reference with an exact valid offset",
                      "answer_boundary_whitespace_trimmed_for_labels": sorted(trimmed_boundaries),
                      "out_of_window_label": "CLS", "invalid_train_answer_policy": invalid_policy if training else None}


def decode_spans(rows, features, start_logits, end_logits, settings):
    if len(features) != len(start_logits) or len(features) != len(end_logits):
        raise ValueError("Feature/logit count mismatch")
    by_id = {r["id"]: r for r in rows}
    candidates, nulls = {}, {}
    for feature, starts, ends in zip(features, start_logits, end_logits):
        starts, ends = np.asarray(starts), np.asarray(ends)
        if starts.shape != (len(feature["offsets"]),) or ends.shape != starts.shape:
            raise ValueError("Logit sequence length mismatch")
        if not np.isfinite(starts).all() or not np.isfinite(ends).all():
            raise ValueError("Non-finite QA logits")
        row_id = feature["id"]
        cls = feature["cls_index"]
        nulls[row_id] = min(nulls.get(row_id, math.inf), float(starts[cls] + ends[cls]))
        valid = [i for i, offset in enumerate(feature["offsets"]) if offset is not None]
        top_starts = sorted(valid, key=lambda i: (-float(starts[i]), i))[:settings.n_best]
        top_ends = sorted(valid, key=lambda i: (-float(ends[i]), i))[:settings.n_best]
        for start in top_starts:
            for end in top_ends:
                if end < start or end - start + 1 > settings.max_answer_length:
                    continue
                char_start, char_end = feature["offsets"][start][0], feature["offsets"][end][1]
                if char_end <= char_start:
                    continue
                value = float(starts[start] + ends[end])
                if row_id not in candidates or value > candidates[row_id]["span_score"]:
                    candidates[row_id] = {"text": by_id[row_id]["context"][char_start:char_end],
                                          "start": char_start, "end": char_end, "span_score": value}
    predictions, details = {}, {}
    for row in rows:
        row_id = row["id"]
        if row_id not in candidates:
            raise ValueError(f"{row_id}: no valid context span; adjust decoding settings")
        best = candidates[row_id]
        null_diff = nulls[row_id] - best["span_score"]
        prediction = "" if settings.allow_no_answer and null_diff > settings.null_score_diff_threshold else best["text"]
        predictions[row_id] = prediction
        details[row_id] = {**best, "null_score": nulls[row_id], "null_score_difference": null_diff}
    return predictions, details
