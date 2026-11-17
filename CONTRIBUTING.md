# Contributing
Work on phase branches (`research/english-baseline`, `research/multilingual-benchmark`, `research/multilingual-finetuning`, `research/indic-aec`, `research/domain-adaptation`, `research/native-indic-qa`, `research/final-evaluation`) and review changes before merging.

Do not overwrite raw translations. Link each correction to its original ID and reviewer. Keep secrets and large checkpoints out of Git. Before handoff run `python -m unittest discover -s tests -v`, validate datasets and include exact commands, revisions and limitations. Do not alter held-out test sets to improve model scores.
