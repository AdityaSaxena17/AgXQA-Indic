# Phase 6 — Build a Native Indic Agricultural QA Dataset

**Owner:** Aditya Saxena
**Status:** Pending. No completed experiments or measured results are claimed.

## Research objective
Create source-grounded QA directly in each Indic language. Human reviewers verify answers and assign document-disjoint splits before any training.

## Inputs and outputs
- Inputs: Native source passages reserved from adaptation/training.
- Deliverables: Native QA benchmark, gold spans, source manifest and review coverage.
- Shared contracts: [schemas](../schemas/README.md), [evaluation protocol](../docs/evaluation.md).

## Starter workflow
From the repository root:
```sh
python -m shared.plan phase_06_native_indic_qa/config.json
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
