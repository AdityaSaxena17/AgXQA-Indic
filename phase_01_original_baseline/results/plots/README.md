# Measured plots and one unavailable measurement

Regenerate with `python phase_01_original_baseline/scripts/generate_plots.py` after installing the phase requirements.

| File | Data source / meaning |
| --- | --- |
| `training_loss.png` | Actual mean training loss for all 20 epochs from `training_logs.csv` |
| `validation_loss.png` | **Availability notice only:** the original run did not record validation loss; no curve or substitute values are plotted |
| `baseline_comparison.png` | Published reference (hatched bars), measured before-fine-tuning initialization, and measured final reimplementation |
| `f1_across_epochs.png` | Actual validation EM and F1 on all 353 validation questions, each epoch |
| `dataset_distribution.png` | Verified official split counts: train 1503, validation 353, test 330 |

XLM-R, general RoBERTa Base, BERT Base, and DistilBERT have no measured scores in this contribution and are not plotted as zero-valued bars. Published reference values are not represented as our own measurements. Scores use SQuAD-style English normalization; see the separate shared-protocol rescore in `../shared_protocol_results.json`.
