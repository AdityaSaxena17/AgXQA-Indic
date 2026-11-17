"""Protocol v1: Unicode NFC/casefold, punctuation to spaces, whitespace tokens."""
from collections import Counter
import unicodedata

def normalize(text):
    text = unicodedata.normalize('NFC', text).casefold()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split())

def exact_match(prediction, reference):
    return float(normalize(prediction) == normalize(reference))

def token_f1(prediction, reference):
    p, r = normalize(prediction).split(), normalize(reference).split()
    if not p or not r:
        return float(p == r)
    overlap = sum((Counter(p) & Counter(r)).values())
    return 2.0 * overlap / (len(p) + len(r))

def best_scores(prediction, references):
    if not references:
        raise ValueError('References must not be empty')
    return {'em': max(exact_match(prediction, r) for r in references),
            'f1': max(token_f1(prediction, r) for r in references)}
