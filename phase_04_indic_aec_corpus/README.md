# Phase 4 — Build Our Own Indic-AEC

**Owner:** Aditya Pise
**Status:** Pending. No completed experiments or measured results are claimed.

## Research objective
Collect native agricultural extension documents with source/license metadata; clean and deduplicate before splitting by source document.

## Inputs and outputs
- Inputs: Permitted native Hindi, Marathi, Bengali and Tamil documents.
- Deliverables: Indic-AEC corpus, provenance, exclusion list and corpus statistics.
- Shared contracts: [schemas](../schemas/README.md), [evaluation protocol](../docs/evaluation.md).

## Starter workflow
From the repository root:
```sh
python -m shared.plan phase_04_indic_aec_corpus/config.json
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
