# Phase 1 checkpoints

**No model weights are committed to GitHub.**

| Checkpoint | Status | Public link |
| --- | --- | --- |
| AgRoBERTa reimplementation, final epoch 20 / step 7900 | Trained, saved locally, reloaded and evaluated on 330 test questions | Not uploaded; no public URL exists yet |
| Authors' original AgRoBERTa QA adapter/head | Missing from the pinned release according to the supplied audit | No usable trained QA checkpoint was evaluated |
| English-trained XLM-R for Experiment 3A | Not trained in this contribution | Pending |

The local AgRoBERTa artifact is an adapter/head checkpoint at the source run's `training/final/` directory. Its files and hashes are recorded in [the checkpoint manifest](../results/provenance/checkpoint_manifest.json). It contains `adapter.safetensors`, `model_head.safetensors`, their configs, tokenizer assets, optimizer state and training metadata. It requires the pinned [agricultural backbone](https://huggingface.co/msu-ceco/roberta-ft-on-agextcorpus-2023-12-10_v2/tree/bdfdd689aede30b4d7d84cb6e267b98a88d58e40).

The source run identifier is `reproduction-2026-11-17-mps`. This repository does not contain that original workspace or its model weights. Another researcher can reproduce the checkpoint using [the training instructions](../README.md#16-steps-to-reproduce); exact cross-device numerical equality is not promised.

Before marking Phase 1 complete, train the English XLM-R baseline, save its complete model/tokenizer/configuration, upload it to an explicitly selected Hugging Face repository or approved shared storage, then replace the pending entry with an immutable revision/link and hashes. Uploading our AgRoBERTa adapter/head would also make the measured result easier to inspect. No upload destination or fabricated checkpoint URL is supplied here.
