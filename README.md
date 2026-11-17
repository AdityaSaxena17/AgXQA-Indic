# AgXQA-Indic

Multilingual agricultural question answering for English, Hindi, Marathi, Bengali and Tamil.

This research scaffold organizes seven experiments to study cross-lingual transfer, multilingual fine-tuning, agricultural domain adaptation and evaluation on native-language QA. It provides working dataset validation and evaluation utilities plus explicit experiment starters. **It does not contain trained models, released research datasets or measured research results.**

## Research questions
1. How much does English-trained QA degrade across Indic languages?
2. Does multilingual fine-tuning reduce that gap?
3. Does Indic-AEC domain adaptation improve performance under controlled conditions?
4. Do gains transfer to independently authored native Indic QA?

## Phases and owners
| Phase | Experiment | Owner | Status |
|---|---|---|---|
| 1 | [Reproduce the Original AgXQA Baseline](phase_01_original_baseline/README.md) | Dhiraj | Pending |
| 2 | [Create AgXQA-Indic](phase_02_multilingual_benchmark/README.md) | Saunak + language reviewers | Pending human review; translations reported generated |
| 3 | [Multilingual Fine-Tuning and Cross-Lingual Evaluation](phase_03_multilingual_finetuning/README.md) | Saunak | Pending |
| 4 | [Build Our Own Indic-AEC](phase_04_indic_aec_corpus/README.md) | Aditya Pise | Pending |
| 5 | [Domain Adaptation Using Indic-AEC](phase_05_domain_adaptation/README.md) | Aditya Pise | Pending |
| 6 | [Build a Native Indic Agricultural QA Dataset](phase_06_native_indic_qa/README.md) | Aditya Saxena | Pending |
| 7 | [Final Comparative Evaluation](phase_07_comparative_evaluation/README.md) | Dhiraj | Pending |

Phase 2 translations were reported generated, possibly through Gemma; neither the exact model nor the files have been verified here. Human review remains pending. Dhiraj's Phase 7 assignment follows the prior roadmap's allocation of the otherwise unspecified second phase.

## Workflow and datasets
Phase 1 establishes the English baseline. Phase 2 produces aligned AgXQA-Indic records for phase 3 (3A English-only training; 3B multilingual training). Phase 4 builds Indic-AEC for phase 5 adaptation. Phase 6 reserves native documents and constructs native QA. Phase 7 compares frozen predictions. Phase 6 test documents must be excluded from phase 4/5 training.

See `data/` dataset cards, `schemas/` contracts, `docs/annotation.md` and `docs/reproducibility.md`. Original AgXQA source, paper citation, license and exact baseline setup must be confirmed before claiming reproduction. No dataset counts are assumed.

## Quick start
Python 3.10+; the shared utilities and tests use only the standard library.
```sh
python -m unittest discover -s tests -v
python -m shared.validate data/examples/qa.jsonl
python -m shared.evaluate data/examples/qa.jsonl data/examples/predictions.jsonl --output /tmp/agxqa-demo.json
python -m shared.plan phase_03_multilingual_finetuning/config.json
```
The tiny example is explicitly synthetic and must never appear in reported research results. See [evaluation definitions](docs/evaluation.md). Training and API-backed generation scripts are intentionally incomplete and fail clearly until implemented. Optional dependency suggestions are in `requirements-training.txt`; no training environment is claimed validated.

## Results — pending
| Approach | EN F1 | HI F1 | MR F1 | BN F1 | TA F1 |
|---|---|---|---|---|---|
| Original English baseline | Pending | N/A | N/A | N/A | N/A |
| XLM-R English-trained | Pending | Pending | Pending | Pending | Pending |
| XLM-R multilingual | Pending | Pending | Pending | Pending | Pending |
| XLM-R + Indic-AEC | Pending | Pending | Pending | Pending | Pending |

EM, native QA results, ablations, uncertainty estimates and error analysis are also pending. No improvement is asserted. `results/status.json` is a tracker, not an experiment output. Store real outputs with dataset/checkpoint hashes and evaluation version.

## Team review and plan
- Aditya Saxena: Hindi review; phase 6.
- Dhiraj: Tamil review; phases 1 and 7.
- Saunak: Bengali review; phases 2 and 3.
- Aditya Pise: Marathi review; phases 4 and 5.

Day 1: freeze schemas/splits, audit translations, acquire sources and baseline setup. Day 2: review data, implement/run baseline and multilingual experiments, build corpus and native QA. Day 3: finish feasible adaptation, integrate predictions and report measured results and unfinished work honestly. This is a relative  plan, not a claim of completion or a newly confirmed calendar deadline.

## Limitations, references and license
Whitespace token F1 is a transparent starter metric with known Indic tokenization limits. Review coverage, translation provenance and data rights need verification. Compute availability may limit the day scope; optional RAG/agentic experiments are deferred. Record verified original AgXQA and model citations in `docs/references.md`. No publication DOI or citation is invented. Code license selection is pending owner decision; third-party data retain their original terms and are not automatically redistributable.

See [contribution workflow](CONTRIBUTING.md).
