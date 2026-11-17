# Phase 1 — Reproduce the Original AgXQA Baseline

**Owner:** Dhiraj
**Status:** Pending. No completed experiments or measured results are claimed.

## Research objective
Acquire the original English dataset and reproduce the published setup. Confirm the paper, model, version, license and official splits before training.

## Inputs and outputs
- Inputs: Official English splits and published methodology.
- Deliverables: Baseline checkpoint, predictions, published-versus-reproduced comparison and deviations.
- Shared contracts: [schemas](../schemas/README.md), [evaluation protocol](../docs/evaluation.md).

## Starter workflow
From the repository root:
```sh
python -m shared.plan phase_01_original_baseline/config.json
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
