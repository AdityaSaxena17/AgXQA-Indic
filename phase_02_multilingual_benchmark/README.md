# Phase 2 — Create AgXQA-Indic

**Owner:** Saunak + language reviewers
**Status:** Translations reported generated; files/model unverified; manual review pending.

## Research objective
Import existing translations; preserve originals and review context, question, terminology and extractive answer spans. Exact translation provider/model is unconfirmed.

## Inputs and outputs
- Inputs: English records and existing Hindi, Marathi, Bengali and Tamil translations.
- Deliverables: Aligned reviewed benchmark, review log, provenance and validation report.
- Shared contracts: [schemas](../schemas/README.md), [evaluation protocol](../docs/evaluation.md).

## Starter workflow
From the repository root:
```sh
python -m shared.plan phase_02_multilingual_benchmark/config.json
python -m shared.validate data/examples/qa.jsonl
python -m shared.evaluate data/examples/qa.jsonl data/examples/predictions.jsonl --output /tmp/agxqa-demo.json
```
The example is synthetic plumbing data, not research evidence. `config.json` is an editable planning template. Phase-specific scripts describe remaining implementation and exit with a nonzero status; they do not train models, call paid APIs or fabricate results.

## Completion checklist
- [ ] Resolve input paths, source permissions, dataset/model revisions and configuration.
- [ ] Implement and review phase-specific experiment steps.
- [ ] Validate IDs, answer offsets and split/document leakage.
- [ ] Record commands, environment, seed, hardware and artifact hashes.
- [ ] Save predictions and actual results with review coverage and limitations.
- [ ] Hand off deliverables to downstream owners; never label a dry run as complete.
