# Phase 2 preparation figures

These figures were generated from the actual source/preparation report, without model inference. `language_distribution.png` displays 2,186 prepared English rows and zero Indic machine candidates; these are not final release counts. `error_distribution.png` displays five English questions flagged for invalid source offsets (nine answer references).

`translation_quality.png`, `human_review_outcomes.png` and `answer_span_validity.png` contain explicit missing-measurement notices. They do not contain illustrative or estimated quality scores. Regenerate with `scripts/dataset_statistics.py --plots` after actual translation/review. Paired before/after figures will disclose the accepted review-resolved subset and excluded unresolved/rejected records.
