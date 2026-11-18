import json
import urllib.request
from pathlib import Path

from .config import COUNTS, DATA_REVISION, DATASET, DATA_SHA256, FILES
from .io import sha256, write_json


def download_data(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REVISION}/{filename}"
        target = directory / filename
        # Always fetch the pinned content; atomic replacement avoids partial inputs.
        temp = target.with_suffix(".download")
        with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temp.replace(target)
    _, report = load_data(directory)
    write_json(directory / "download-manifest.json", report)
    return report


def load_data(directory):
    directory = Path(directory)
    splits, seen, reports = {}, set(), {}
    for split, filename in FILES.items():
        path = directory / filename
        checksum = sha256(path)
        if checksum != DATA_SHA256[split]:
            raise ValueError(f"{split}: SHA-256 differs from pinned published file: {checksum}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != COUNTS[split]:
            raise ValueError(f"{split}: expected {COUNTS[split]} rows, found {len(rows)}")
        bad_offsets, answerless = [], []
        for row in rows:
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"{split}: ID must be a nonempty string")
            if row["id"] in seen:
                raise ValueError(f"Duplicate ID across published splits: {row['id']}")
            seen.add(row["id"])
            if not all(isinstance(row.get(k), str) for k in ("question", "context")):
                raise ValueError(f"{row['id']}: question/context must be strings")
            answers = row.get("answers", {})
            texts, starts = answers.get("text"), answers.get("answer_start")
            if not isinstance(texts, list) or not isinstance(starts, list) or len(texts) != len(starts):
                raise ValueError(f"{row['id']}: answers require parallel text and answer_start arrays")
            if not texts:
                answerless.append(row["id"])
            for index, (text, start) in enumerate(zip(texts, starts)):
                if not isinstance(text, str) or not text or type(start) is not int:
                    raise ValueError(f"{row['id']}: invalid answer schema")
                if start < 0 or row["context"][start:start + len(text)] != text:
                    bad_offsets.append({"id": row["id"], "answer_index": index,
                                        "answer_start": start, "text": text})
        if split == "test" and answerless:
            raise ValueError("The pinned test set should have gold answers for every row")
        splits[split] = rows
        reports[split] = {"path": str(path.resolve()), "sha256": checksum, "pinned_hash_verified": True, "rows": len(rows),
                          "multiple_reference_rows": sum(len(r["answers"]["text"]) > 1 for r in rows),
                          "answerless_ids": answerless, "invalid_offsets": bad_offsets,
                          "invalid_offset_rows": len({item["id"] for item in bad_offsets})}
    train_contexts = {r["context"] for r in splits["train"]}
    train_pairs = {(r["question"], r["context"]) for r in splits["train"]}
    reused = [r["id"] for r in splits["test"] if r["context"] in train_contexts]
    duplicate = [r["id"] for r in splits["test"] if (r["question"], r["context"]) in train_pairs]
    return splits, {"dataset": DATASET, "requested_revision": DATA_REVISION,
                    "local_content_note": "All local file hashes match the pinned published files.",
                    "splits": reports, "train_test_overlap": {"verbatim_context_rows": len(reused),
                    "test_ids_with_train_context": reused, "exact_question_context_rows": len(duplicate),
                    "test_ids_with_train_pair": duplicate}, "test_offsets_modified": False}


def subset(rows, limit):
    if limit is not None and limit < 1:
        raise ValueError("Row limit must be positive")
    return rows if limit is None else rows[:limit]
