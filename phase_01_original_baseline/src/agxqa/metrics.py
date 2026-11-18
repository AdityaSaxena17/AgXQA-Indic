"""SQuAD 1.1 normalization, multiset token F1, and best-reference scoring."""
import re
import string
from collections import Counter


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction, reference):
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction, reference):
    predicted, gold = normalize(prediction).split(), normalize(reference).split()
    common = sum((Counter(predicted) & Counter(gold)).values())
    if not common:
        return 0.0
    precision, recall = common / len(predicted), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def score(rows, predictions):
    expected, actual = {r["id"] for r in rows}, set(predictions)
    if len(expected) != len(rows):
        raise ValueError("Gold IDs must be unique")
    if not rows:
        raise ValueError("Cannot score an empty split")
    if expected != actual:
        raise ValueError(f"Prediction coverage mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    details = []
    for row in rows:
        prediction = predictions[row["id"]]
        if not isinstance(prediction, str):
            raise ValueError(f"{row['id']}: prediction must be text")
        references = row["answers"]["text"] or [""]
        details.append({"id": row["id"], "prediction": prediction, "references": references,
                        "exact_match": max(exact_match(prediction, ref) for ref in references),
                        "f1": max(token_f1(prediction, ref) for ref in references)})
    return {"exact_match": 100 * sum(r["exact_match"] for r in details) / len(rows),
            "f1": 100 * sum(r["f1"] for r in details) / len(rows),
            "total": len(rows), "prediction_coverage": len(predictions),
            "scorer": "SQuAD 1.1; normalized EM; multiset token F1; independent best reference per metric"}, details
