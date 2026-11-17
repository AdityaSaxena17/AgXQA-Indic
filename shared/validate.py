"""Validate normalized extractive QA JSONL and within-file split integrity."""
import argparse
import json
from pathlib import Path

LANGUAGES = {'en', 'hi', 'mr', 'bn', 'ta'}

def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{number}: invalid JSON') from exc
    return rows

def validate(rows):
    if not rows:
        raise ValueError('Dataset is empty')
    seen, groups, documents = set(), {}, {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Record must be an object')
        for field in ('id', 'language', 'split', 'context', 'question', 'source_document_id'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Missing/invalid {field}')
        if row['language'] not in LANGUAGES or row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('Unsupported language or split')
        key = (row['id'], row['language'])
        if key in seen:
            raise ValueError(f'Duplicate ID/language: {key}')
        seen.add(key)
        for mapping, group in ((groups, row['id']), (documents, row['source_document_id'])):
            if group in mapping and mapping[group] != row['split']:
                raise ValueError(f'Split leakage: {group}')
            mapping[group] = row['split']
        answers = row.get('answers')
        if not isinstance(answers, list) or not answers:
            raise ValueError(f'{key}: at least one answer required')
        for answer in answers:
            if not isinstance(answer, dict):
                raise ValueError('Answer must be an object')
            text, start = answer.get('text'), answer.get('answer_start')
            if not isinstance(text, str) or not text.strip() or type(start) is not int or start < 0:
                raise ValueError(f'{key}: invalid answer')
            if row['context'][start:start + len(text)] != text:
                raise ValueError(f'{key}: answer span mismatch')
    return {'records': len(rows), 'question_groups': len(groups), 'languages': sorted({r['language'] for r in rows})}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    args = parser.parse_args()
    try:
        print(json.dumps(validate(read_jsonl(args.dataset)), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')

if __name__ == '__main__':
    main()
