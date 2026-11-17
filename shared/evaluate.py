"""Evaluate complete prediction coverage for one dataset split."""
import argparse
import hashlib
import json
from pathlib import Path
from shared.metrics import best_scores
from shared.validate import read_jsonl, validate

def evaluate(gold, predictions):
    validate(gold)
    if len({r['split'] for r in gold}) != 1:
        raise ValueError('Evaluate exactly one split at a time')
    lookup = {}
    for row in predictions:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ('id', 'language', 'prediction_text')):
            raise ValueError('Invalid prediction record')
        key = (row['id'], row['language'])
        if key in lookup:
            raise ValueError(f'Duplicate prediction: {key}')
        lookup[key] = row['prediction_text']
    expected = {(r['id'], r['language']) for r in gold}
    if set(lookup) != expected:
        raise ValueError(f'Prediction coverage mismatch: missing={len(expected-set(lookup))}, extra={len(set(lookup)-expected)}')
    scored = []
    for row in gold:
        scored.append({'id': row['id'], 'language': row['language'], **best_scores(lookup[(row['id'], row['language'])], [a['text'] for a in row['answers']])})
    def mean(rows, metric):
        return 100 * sum(r[metric] for r in rows) / len(rows)
    languages = sorted({r['language'] for r in scored})
    by_language = {lang: {'count': sum(r['language'] == lang for r in scored), **{m: mean([r for r in scored if r['language'] == lang], m) for m in ('em', 'f1')}} for lang in languages}
    groups = {}
    for row in scored:
        groups.setdefault(row['id'], {})[row['language']] = row
    complete = [g for g in groups.values() if set(g) == set(languages)] if len(languages) > 1 else []
    # Matched IDs only: avoid attributing different sample composition to language.
    gaps = {}
    if 'en' in languages:
        for lang in languages:
            if lang == 'en':
                continue
            paired = [g for g in groups.values() if 'en' in g and lang in g]
            gaps[lang] = {'paired_count': len(paired), 'f1_gap_points': 100 * sum(g['en']['f1']-g[lang]['f1'] for g in paired)/len(paired) if paired else None}
    return {'protocol': 'agxqa-indic-v1', 'count': len(scored), 'split': gold[0]['split'],
            'by_language': by_language,
            'micro': {m: mean(scored, m) for m in ('em', 'f1')},
            'macro': {m: sum(v[m] for v in by_language.values())/len(languages) for m in ('em', 'f1')},
            'language_gap': gaps,
            'consistency': {'definition': 'all_languages_exactly_correct', 'languages': languages, 'complete_groups': len(complete), 'excluded_groups': len(groups)-len(complete), 'percent': 100*sum(all(r['em'] == 1 for r in g.values()) for g in complete)/len(complete) if complete else None},
            'per_example': scored}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gold')
    parser.add_argument('predictions')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = evaluate(read_jsonl(args.gold), read_jsonl(args.predictions))
        report['input_sha256'] = {k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in [('gold', args.gold), ('predictions', args.predictions)]}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f'Wrote {output}')
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')

if __name__ == '__main__':
    main()
