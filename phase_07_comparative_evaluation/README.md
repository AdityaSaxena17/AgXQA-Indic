# Phase 7 — Final Comparative Evaluation

**Owner:** Dhiraj
**Status:** Pending. No completed experiments or measured results are claimed.

## Research objective
Compare the original English baseline, English-trained XLM-R, multilingual XLM-R and adapted XLM-R on identical eligible test records. RAG/agentic systems are optional future work.

## Inputs and outputs
- Inputs: Frozen gold datasets and model predictions from phases 1, 3 and 5.
- Deliverables: Per-language EM/F1, language gap, aligned correctness consistency, error analysis and tables.
- Shared contracts: [schemas](../schemas/README.md), [evaluation protocol](../docs/evaluation.md).

## Starter workflow
From the repository root:
```sh
python -m shared.plan phase_07_comparative_evaluation/config.json
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
