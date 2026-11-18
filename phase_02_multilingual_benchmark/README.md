# Phase 2 — AgXQA-Indic Benchmark Construction

**Owner:** Saunak. **Preparation snapshot: 18 November 2026.** The English source audit and implementation are available. **IndicTrans2 translation has not run; zero translated QA records and zero human judgments have been produced.** The full multilingual benchmark and Phase 3/5 handoff remain pending. See [completion tracker](results/completion_tracker.json).

The user selected **IndicTrans2 English→Indic, distilled 200M**. Earlier planning notes suggested Gemma/Gemini translations might exist; no such files or provenance were supplied or found in this checkout. Their settings and results are unknown. This implementation records a new local IndicTrans2 workflow. It does not retrospectively label any earlier output.

## 1. Objective

Extend English AgXQA to Hindi, Marathi, Bengali and Tamil while preserving question identity, agricultural meaning, every gold-answer reference and official split membership. Deliver a validated parallel benchmark for extractive QA, with immutable machine output and traceable human corrections. A generated translation is a candidate, not evidence of human validation.

## 2. Original Dataset Description

Source: [`msu-ceco/agxqa_v1`](https://huggingface.co/datasets/msu-ceco/agxqa_v1), pinned to **`9fb0cf2a6d03ccfccbd67d5966179b43b5b77e2c`**. The task is English agricultural/irrigation extractive question answering. Each source JSONL record contains `id`, `context`, `question`, parallel `answers.text`/`answers.answer_start` lists, `category` and `references`.

The three `agxqa-{train,validation,test}-2024-06-11.jsonl` files are verified against fixed SHA-256 values before use. [source_preflight.json](results/source_preflight.json) records the hashes and measured audit. The official files are never edited. Source inspection found **391 distinct contexts, 2,185 distinct question strings and 2,045 distinct answer strings** across 2,186 unique question IDs. Repeated text is cached during translation without changing question counts.

## 3. Supported Languages

| Language | Code | IndicTrans2 target tag | Train target | Validation target | Test target | Total target |
|---|---|---|---:|---:|---:|---:|
| English | en | Source `eng_Latn` | 1,503 | 353 | 330 | 2,186 |
| Hindi | hi | `hin_Deva` | 1,503 | 353 | 330 | 2,186 |
| Marathi | mr | `mar_Deva` | 1,503 | 353 | 330 | 2,186 |
| Bengali | bn | `ben_Beng` | 1,503 | 353 | 330 | 2,186 |
| Tamil | ta | `tam_Taml` | 1,503 | 353 | 330 | 2,186 |
| All languages | | | 7,515 | 1,765 | 1,650 | 10,930 |

English counts were verified from downloaded bytes. Indic counts are **targets**, comprising 8,744 translated QA records. Currently no translated or finalized records exist.

## 4. Translation Methodology

The checkpoint is a sequence-to-sequence translation model, accessed locally with Transformers and IndicTransToolkit. It uses source/target language conditioning, not a chat prompt. [translation_prompt.txt](prompts/translation_prompt.txt) records the exact planned conditioning procedure; chat prompt, API sampling temperature, paid API token consumption and API cost are not applicable (`null`). Execution duration is unmeasured until a run occurs.

Translate each context, question and **all** answer strings independently. The same context is translated once per target language/configuration/runtime and reused through a SQLite cache. Sentence segmentation followed by whitespace segmentation keeps actual preprocessed tokenizer inputs within 256 tokens. No input truncation is allowed. This retains normalized source words, but chunk boundaries can reduce discourse coherence; reviewers must inspect full paragraphs. Answers and questions can also be chunked if they exceed the budget.

Each segment records the normalized English text, preprocessed input, decoded model output, generated token IDs, postprocessed output, source/output token counts and length-cap flags. Joined translations and field-level records are appended to language JSONL files. NFC and whitespace cleanup happen in separate candidates, leaving raw records unchanged. Target generations reaching the cap are flagged for review, rather than assumed complete in meaning.

Answer alignment accepts a **single exact substring match** in the cleaned context. Missing or repeated matches retain their original answer index and all possible positions in `answer_alignment`, and require human correction. The pipeline neither picks an arbitrary repeated occurrence nor appends an answer to a context. Inflection and paraphrase can make independent answer translation difficult; the pilot will measure this limitation before full execution.

Unit translations are retried at most three times, with one- and two-second delays between attempts. Failed QA records remain in raw logs. `--resume` requires identical configuration, input selection, pipeline source hashes and runtime metadata; `--retry-failed` appends a numbered retry. Successful raw records cannot be replaced. A malformed interrupted JSONL tail fails with an error and must be preserved before explicit recovery. Model/runtime changes require a separate raw run.

## 5. Translation Model and Configuration

| Setting | Prepared value |
|---|---|
| Checkpoint | `ai4bharat/indictrans2-en-indic-dist-200M` |
| Immutable revision | `2cfd0a20057756d756bf3e53528f492afc52f91a` |
| Provider | Local Hugging Face Transformers |
| Toolkit | IndicTransToolkit 1.1.1 |
| Decoding | Beam search, 5 beams, one output, `do_sample=false` |
| Input / output cap | 256 actual preprocessed input tokens / `max_length=256` output tokens |
| Runtime | Float32, eager attention, 4 CPU threads |
| Device | Auto: CUDA, then Apple MPS, then CPU; explicitly selectable |
| Batch size | 4 **segments from one translation unit**, not four QA records |
| Seed | 42; exact reproducibility across devices is not guaranteed |
| Remote model code | `trust_remote_code=true`, pinned model/code revision |

[translation_config.yaml](configs/translation_config.yaml) and [config.json](config.json) contain identical settings. The default CLI uses JSON so non-model checks need only Python's standard library. Requirements declare a candidate environment; **the full translation dependency environment and neural integration have not been executed or validated**. Run manifests will record actual package versions, Python/platform/device, invocation, code/input hashes and measured per-session/total execution seconds. They do not estimate unrecorded costs.

The [official model](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) is gated with automatic access after the account accepts its sharing conditions. No local Hugging Face token was configured at access preflight. The user must accept the conditions and authenticate locally; credentials must not be pasted into issues or committed. Model weights have not been downloaded by this phase. Downloading with `scripts/download_model.py` records file hashes without executing remote Python; inspect that pinned source before inference. The Mac/MPS path is a proposed setting, not a measured throughput or compatibility claim.

## 6. Dataset Construction Pipeline

`Pinned English bytes → immutable per-language MT logs → NFC candidates and exact answer alignment → complete automated audit → native-language review and source adjudication → correction revalidation → gated final build → reports and handoff`.

| Script | Behavior |
|---|---|
| `prepare_source.py` | Download/verify pinned source, audit counts, offsets and source overlap |
| `download_model.py` | Download authorized checkpoint and hash files; no neural execution |
| `translate.py` | Print plan by default; inference requires `--run-model` and an explicit raw directory |
| `clean_translations.py` | Make a new candidate version, preserve raw logs, align answers and run QC |
| `validate_schema.py` | Schema, record coverage, IDs and split checks |
| `validate_answer_spans.py` | Exact character offsets and all-reference completeness |
| `validate_language.py` | Report stored script/language-ID assessments from cleaning |
| `check_data_leakage.py` | Audit source overlap and reject changed question split membership |
| `create_review_worksheets.py` | Create seeded/flagged review requests, never completed judgments |
| `merge_languages.py`, `build_dataset.py` | Same full five-language build gate; refuse incomplete/rejected/unreviewed required records |
| `dataset_statistics.py` | Actual counts, quality/coverage reports and optional plots; no neural execution |

The build writes no final files when requirements fail. Raw-output, cleaned-candidate and source hashes bind corrections to the exact reviewed version; changed candidate text invalidates an earlier judgment. Final directories must be new/empty; files are staged and validated before publication. Existing official document overlap is retained and reported. `human_reviewed/*.jsonl` contains only actual reviewed accepted records; unsampled, unflagged train/validation rows may be `automated_accepted`, with `human_reviewed=false`.

## 7. Dataset Schema

Three views are produced from the same accepted records:

- `data/agxqa_indic/final/{train,validation,test}.jsonl`: flattened SQuAD-style records. `id` is `<original_id>_<language>`, `original_id` is the unchanged source ID, `answers` contains parallel `text` and `answer_start` lists, and records include language, split, source-document identity, validation status, `human_reviewed`, source/raw hashes and review provenance.
- `data/agxqa_indic/final/parallel_multilingual.jsonl`: one group per unchanged `original_id`, one official split and a `languages` object containing all five records.
- `data/agxqa_indic/final/shared/{train,validation,test}.jsonl`: compatibility view for this repository's shared validator/evaluator. `id` is the common source ID and `answers` is a list of `{text, answer_start}` objects. Use these files with `python -m shared.validate`.

Do not feed the primary flattened view directly to the shared validator: its answer/ID convention differs. Python character offsets must satisfy `context[start:start + len(text)] == text`; byte offsets are invalid. All source answer references must be represented. The builder blocks unresolved source gold errors rather than silently deleting references.

## 8. Automated Validation

All generated records require schema, expected `(original_id, language)` coverage, duplicate-key, ID alignment, split, source hash, exact nonempty answer span, reference count and Unicode checks. Indic text uses NFC; replacement characters, surrogates and unexpected control characters are flagged, with meaningful joiners preserved. Missing/ambiguous spans are review failures, not fuzzy automatic matches.

Language QC combines expected-script ratios, long English-passage flags and the non-neural character n-gram classifier `langid==1.1.6`, restricted to the five languages. A script check alone cannot distinguish Hindi and Marathi. Classifier scores are not calibrated probabilities and are not semantic evaluation. `--skip-language-id` explicitly flags every candidate for human language verification. `validate_language.py` reads stored assessments; rerun cleaning into a new version to recompute them.

Numbers are compared as multisets after decimal-digit normalization; a small documented unit alias lexicon flags missing units. These conservative heuristics can miss number words or equivalent conversions and can overflag valid forms. Native reviewers must resolve flagged cases explicitly. There is no automatic claim of semantic equivalence, correct agricultural terminology or absence of semantic near duplicates.

## 9. Human Annotation Guidelines

Follow [annotation_guidelines.md](human_review/annotation_guidelines.md). Review context/question meaning, fluency, agricultural terminology, numbers/units and every gold answer. Score the original candidate 1–5 on semantic equivalence, fluency, terminology and answer correctness, then decide Accept / Correct / Reject. Store original outputs separately from corrections, with actual reviewer identity, timestamp, raw hash and rationale.

| Reviewer | Language | Prepared minimum target |
|---|---|---|
| Aditya Saxena | Hindi | 442 questions plus additional flagged questions |
| Aditya Pise | Marathi | Same |
| Saunak | Bengali | Same |
| Dhiraj | Tamil | Same |

Proposed sampling: 5% train (76), 10% validation (36), 100% test (330), seed 42, per language; the percentages are implementation choices, not requirements inferred from the original paper. Every flagged record also requires review. Current actual review coverage is **0%**, with four header-only target-language CSVs and five prefilled **unjudged** English source-review requests. Assignments are not completed annotations. Resolving English source errors requires real source adjudication in addition to Indic review.

## 10. Translation Quality Evaluation

Report successful translated QA records / expected translated QA records as completion rate. Report all-reference valid-span records / observed candidates as raw span validity; failed/missing coverage is separately visible. Human means and Accept/Correct/Reject rates use actual submitted judgments, while coverage counts unique question IDs over expected IDs by language/split. Oversampling flagged records biases these means and must be disclosed.

Cohen's kappa is computed only for explicitly independent reviewers of the same language/examples, per reviewer pair, with at least two paired observations. It is `null` when absent or mathematically undefined. The four assigned people reviewing different languages do not provide inter-annotator agreement. Before/after span comparisons use matched accepted review-resolved IDs, disclose exclusions, and cannot be presented as full-dataset quality. Human error categories are reported separately from automated issue codes. No BLEU/COMET or semantic scores are invented without suitable reference data and an explicit evaluation protocol.

## 11. Dataset Statistics

| Observation | Actual preparation snapshot |
|---|---:|
| Verified source QA records | 2,186 |
| Distinct source contexts | 391 |
| Translated QA records | 0 |
| Completed human judgments | 0 |
| Final multilingual records emitted | 0 |
| Source references with invalid offsets | 9 across 5 test questions |
| Cross-split reference-document groups | 39 |
| Cross-split exact-context groups | 207 |

See [dataset_statistics.json](results/dataset_statistics.json), [translation_report.json](results/translation_report.json) and shared [statistics.json](../data/agxqa_indic/statistics.json). Source/preparation counts must not be relabeled as released final dataset counts. Unmeasured quality means, span rates with no candidates, kappa and execution duration are `null` or absent, not synthetic estimates.

## 12. Error Analysis

The source questions `1170477`, `1174212`, `1183012`, `1181597` and `481` contain invalid answer offsets. `481` includes a placeholder-like gold string. The builder requires explicit English source review and keeps the original files/Phase 1 experiment intact. A corrected Phase 2 English benchmark is a new annotation version, requiring version disclosure in downstream comparisons.

No translation-error taxonomy counts have yet been measured. [error_analysis.csv](results/error_analysis.csv) currently reports only actual English source flags. Human terminology, answer mismatch, grammar, meaning change, missing content and numerical errors can be entered in the review CSV's `error_categories` column. Analysis must distinguish alignment failure from incorrect translation and language heuristics from human-confirmed errors.

## 13. Data Leakage Prevention

All five variants of an original question retain its official split. Question IDs crossing splits are rejected. `source_document_id` is derived from source reference URL(s), with citation text as a labeled fallback. These group IDs are not split-prefixed. They are practical citation groups, not a verified bibliography of unique documents; URL-set identity can miss partial overlaps and semantic duplicates.

The official dataset already overlaps across documents and verbatim contexts: **39 citation groups and 207 exact context groups** cross splits. We preserve this for baseline comparability and disclose it, rather than claim document-disjoint evaluation. Shared validation is therefore run one official split at a time. A new document-disjoint benchmark would require a separately versioned experiment and new baselines. Translation caching across splits is deterministic reuse of the same frozen MT output, not training on test answers. Human test review must remain independent of model performance.

## 14. Results and Visualizations

Files under `results/` include the source preflight, per-record automated audit, header-only human evaluation table, actual source issue counts, machine-readable completion tracker and five required plots. At this stage `language_distribution.png` shows prepared English/source counts, `error_distribution.png` shows source issue flags, and translation-quality/span-correction/review-outcome figures explicitly state that observations are unavailable. No illustrative scores from the brief are used.

Preparation verification: **39 repository tests passed**, including 22 Phase 2 tests using synthetic data/fake translation backends and no neural execution. Shared synthetic validation/evaluation also passed. An actual source-only build preflight exited 2 and emitted no final files, correctly blocking missing translations and required source judgments. See [verification.json](results/verification.json); checkpoint loading, real MT, language-classifier integration and human review remain unrun.

After an actual run and review, regenerate reports from the preserved candidates and completed worksheets. The paired span plot will use the same accepted review-resolved IDs before/after; raw overall candidate validity and unresolved records remain in the report. This preparation is not a completed Phase 2 submission.

## 15. Limitations

The selected model is gated and the Mac neural path remains untested. Model code is loaded from the pinned checkpoint using `trust_remote_code`; inspect downloaded files first. Chunked independent field translation can lose coherence or produce answers that do not match the context; the pilot and native review are essential. Heuristics cannot certify semantics, fluency, terminology or unit equivalence. Original source offsets and document overlap need explicit handling. Human review, multilingual generation, final build, rights/redistribution review and downstream handoff remain pending. Phase 1's English-trained XLM-R model is also still pending; Phase 2 does not satisfy that separate prerequisite for Experiment 3A.

## 16. Reproducibility Instructions

Run from the repository root, using Python 3.11 on this Mac. Keep translation dependencies separate from the Phase 1 environment. The installation below is a **recipe**, not a claim it was executed here:

```sh
python3.11 -m venv phase_02_multilingual_benchmark/.venv
phase_02_multilingual_benchmark/.venv/bin/python -m pip install -r phase_02_multilingual_benchmark/requirements.txt
mkdir -p phase_02_multilingual_benchmark/outputs
phase_02_multilingual_benchmark/.venv/bin/python -m pip freeze > phase_02_multilingual_benchmark/outputs/environment.freeze.txt
```

The declared pins are not a full transitive dependency lock; save the resolved environment and run manifest. Standard-library synthetic checks do not need translation dependencies:

```sh
python3 -m unittest discover -s tests -v
python3 phase_02_multilingual_benchmark/scripts/prepare_source.py --download
python3 phase_02_multilingual_benchmark/scripts/translate.py --split validation --limit 5
```

The last command prints the plan for **five English validation questions × four languages = 20 translated QA records** and loads no neural model. To reuse already downloaded English files, append `--source-dir /path/to/english-files` to each script. The default local source directory is ignored by Git. This preparation audit used the existing verified English files rather than downloading another dataset copy.

Before model execution, the user must approve the run, accept the [checkpoint sharing conditions](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M), and authenticate locally using the environment's `hf auth login`. Do not send a token through chat. Download and inspect the pinned checkpoint without executing its Python:

```sh
phase_02_multilingual_benchmark/.venv/bin/hf auth login
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/download_model.py --output-dir phase_02_multilingual_benchmark/.cache/indictrans2-en-indic-dist-200M
```

**Only after explicit approval**, execute the pilot:

```sh
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/translate.py --run-model --split validation --limit 5 --device mps --local-model phase_02_multilingual_benchmark/.cache/indictrans2-en-indic-dist-200M --output-dir phase_02_multilingual_benchmark/outputs/pilot/raw
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/clean_translations.py --raw-dir phase_02_multilingual_benchmark/outputs/pilot/raw --output-dir phase_02_multilingual_benchmark/outputs/pilot/candidates
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/dataset_statistics.py --raw-dir phase_02_multilingual_benchmark/outputs/pilot/raw --candidates-dir phase_02_multilingual_benchmark/outputs/pilot/candidates --results-dir phase_02_multilingual_benchmark/outputs/pilot/results --plots
```

Inspect success/failure counts, output-length flags, alignment failures, sample fidelity and actual speed/memory. A pilot's 20 target rows are not a complete benchmark. Do not extrapolate quality from five questions. If MPS is unsupported, investigate and obtain approval for the revised CPU/runtime plan; do not mix devices in an existing raw run.

The full-run commands below are **prepared and unexecuted**; running all splits needs its own approval after the pilot:

```sh
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/translate.py --run-model --split all --all-rows --device mps --local-model phase_02_multilingual_benchmark/.cache/indictrans2-en-indic-dist-200M --output-dir data/agxqa_indic/raw_translations
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/clean_translations.py
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/validate_schema.py
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/validate_answer_spans.py
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/validate_language.py
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/check_data_leakage.py
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/create_review_worksheets.py --review-dir phase_02_multilingual_benchmark/outputs/review_v1
```

Complete the review CSVs according to the annotation guidelines, preserving raw records. Then:

```sh
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/build_dataset.py --review-dir phase_02_multilingual_benchmark/outputs/review_v1
phase_02_multilingual_benchmark/.venv/bin/python phase_02_multilingual_benchmark/scripts/dataset_statistics.py --review-dir phase_02_multilingual_benchmark/outputs/review_v1 --plots --dataset-statistics-output data/agxqa_indic/statistics.json
python3 -m shared.validate data/agxqa_indic/final/shared/train.jsonl
python3 -m shared.validate data/agxqa_indic/final/shared/validation.jsonl
python3 -m shared.validate data/agxqa_indic/final/shared/test.jsonl
```

Validation scripts exit nonzero for missing full-benchmark coverage or failed checks; this is expected on a pilot or the present zero-translation snapshot. Until all required translations, source corrections and reviews exist, the build exits nonzero and writes a blocking report, with no final benchmark. Preserve/version earlier reports before regenerating them for a new candidate/review snapshot. A finished handoff requires final file hashes, license/source documentation, split counts, full review coverage disclosure, benchmark version and downstream owners' acknowledgment. Never commit authentication secrets, caches or large model weights. Raw research data publication requires checking source redistribution terms.

## 17. Contributors

Saunak owns Phase 2, translation methodology, Bengali review and integration. Aditya Saxena is assigned Hindi review, Aditya Pise Marathi review and Dhiraj Tamil review. These assignments come from the project brief; this snapshot does not claim any of them has performed a review. No external messages or reviewer requests were sent.

## 18. References

- [AgXQA dataset, pinned revision](https://huggingface.co/datasets/msu-ceco/agxqa_v1/tree/9fb0cf2a6d03ccfccbd67d5966179b43b5b77e2c).
- [Original AgXQA authors' repository](https://github.com/MSU-CECO/agxqa_benchmark_v1).
- [IndicTrans2 distilled English→Indic 200M checkpoint](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M/tree/2cfd0a20057756d756bf3e53528f492afc52f91a).
- [IndicTrans2 official inference example, inspected source revision](https://github.com/AI4Bharat/IndicTrans2/blob/4f08e39cc6bf13cd62e2445dc725f22bff1a9219/huggingface_interface/example.py).
- [IndicTransToolkit author repository, inspected source revision](https://github.com/VarunGumma/IndicTransToolkit/tree/3efb8418d0721b4ce267c2b3586899d313191357) and [1.1.1 package](https://pypi.org/project/indictranstoolkit/1.1.1/).
- [langid.py author repository](https://github.com/saffsd/langid.py).
- [Phase 1 measured baseline and source limitations](../phase_01_original_baseline/README.md), [shared evaluation protocol](../docs/evaluation.md), [draft dataset card](../data/agxqa_indic/dataset_card.md).
