# Phase 5 — Domain Adaptation Using Indic-AEC

**Owner:** Aditya Pise
**Status:** Pending. No completed experiments or measured results are claimed.

## Research objective
Continue XLM-R masked-language pretraining on Indic-AEC, then fine-tune QA with the same data and settings as 3B. Exclude native test documents before adaptation.

## Inputs and outputs
- Inputs: Phase 4 training corpus and phase 3 QA training configuration.
- Deliverables: MLM and QA checkpoints, losses, predictions and controlled comparison.
- Shared contracts: [schemas](../schemas/README.md), [evaluation protocol](../docs/evaluation.md).

## Starter workflow
From the repository root:
```sh
python -m shared.plan phase_05_domain_adaptation/config.json
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
