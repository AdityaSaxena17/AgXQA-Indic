# AgXQA-Indic — draft dataset card

**Status: preparation only, 18 November 2026. No translated or finalized multilingual research data released.** Verified source records: 2,186 English QA pairs. Translated QA records: 0. Completed human judgments: 0. Final benchmark records: 0. The target of 10,930 records is not an observed release count. See [actual preparation statistics](statistics.json) and [Phase 2 documentation](../../phase_02_multilingual_benchmark/README.md).

Source: [`msu-ceco/agxqa_v1`](https://huggingface.co/datasets/msu-ceco/agxqa_v1), revision `9fb0cf2a6d03ccfccbd67d5966179b43b5b77e2c`. File hashes and actual source defects are in [source_preflight.json](../../phase_02_multilingual_benchmark/results/source_preflight.json). Raw official source bytes remain unchanged. Language: English; target additions: Hindi, Marathi, Bengali, Tamil. Domain: agriculture/irrigation; task: extractive question answering.

| Planned language coverage | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Each of en/hi/mr/bn/ta | 1,503 | 353 | 330 | 2,186 |
| Five-language target | 7,515 | 1,765 | 1,650 | 10,930 |

English counts are verified. Indic counts are targets. Preserve original IDs and official splits, with all language variants grouped under `original_id`. Primary final files use SQuAD-style parallel answer lists and language-suffixed IDs; `final/shared/` uses the repository's common-ID/list-of-answers contract. `parallel_multilingual.jsonl` groups all five languages. No final files exist yet.

Planned translation: local `ai4bharat/indictrans2-en-indic-dist-200M`, revision `2cfd0a20057756d756bf3e53528f492afc52f91a`, IndicTransToolkit 1.1.1, beam 5, float32, seed 42. No chat API prompt/temperature applies. No earlier Gemma/Gemini provenance is verified. The gated checkpoint has not been downloaded or run here. Raw generation logs remain immutable; cleaned candidates and reviewer corrections are separate versions. Full methodology/configuration and commands are documented in Phase 2.

Human review target: all test questions; seeded 5% train and 10% validation samples; all flagged candidates; separate English source adjudication. Actual coverage is currently zero. Assigned reviewers: Aditya Saxena (Hindi), Aditya Pise (Marathi), Saunak (Bengali), Dhiraj (Tamil). Ratings, identity, decisions and timestamps must come from actual reviewers. No inter-annotator agreement is claimed.

Known limitations: nine invalid English gold offsets across five test questions, including a placeholder-like gold string; 39 citation groups and 207 exact-context groups cross official splits. Preserve this overlap for baseline comparability and disclose it. Citation-based document groups are approximate and do not constitute semantic deduplication. Independent MT answer translation may not yield extractable spans; unresolved cases block final construction. Test annotation changes must be versioned separately from Phase 1's original-source results.

Intended use: controlled multilingual agricultural QA research and cross-lingual consistency analysis. Quality metrics, exclusions, final release version, hosted data URL, file hashes, actual execution metadata and downstream handoff remain pending. Do not infer field reliability or agricultural advice quality from an untranslated/unreviewed preparation snapshot. Keep independently authored native test documents out of later adaptation corpora.

Rights: data retain upstream terms. Dataset/source-document redistribution rights and the final dataset license require verification before publishing research JSONL. The model card's MIT license is not evidence that source agricultural documents can be redistributed. No credential or model weight is included. Synthetic test fixtures live separately under `tests/` and `data/examples/` and never count as research data.
