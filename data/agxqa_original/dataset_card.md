# Original English AgXQA

The Phase 1 download/validation code uses `msu-ceco/agxqa_v1` at revision `9fb0cf2a6d03ccfccbd67d5966179b43b5b77e2c`. The downloaded published files were verified to contain **1,503 train / 353 validation / 330 test rows (2,186 total)**. No raw research data is committed here.

See [Phase 1 dataset statistics and file hashes](../../phase_01_original_baseline/results/dataset_statistics.json), [download instructions](../../phase_01_original_baseline/README.md#16-steps-to-reproduce), and the [upstream dataset](https://huggingface.co/datasets/msu-ceco/agxqa_v1/tree/9fb0cf2a6d03ccfccbd67d5966179b43b5b77e2c). Data retain the original source terms; redistribution rights are not granted by this repository's code-license decision.

The native published schema uses parallel answer-text/start arrays. Its test split has 59 multi-reference rows, nine bad answer offsets across five rows, 309 contexts also present verbatim in train, and one repeated question-context pair. Phase 1 preserves these files and scores answer strings. Adapting raw records to the repository's normalized shared contract requires explicit provenance/document-ID decisions and review; no silent span repair was done.

The original paper-comparison scorer and repository shared scorer differ. Their separately labelled results and the remaining document-disjoint/generalization limits are documented in the Phase 1 README. Synthetic examples in `data/examples/` are not experimental evidence.
