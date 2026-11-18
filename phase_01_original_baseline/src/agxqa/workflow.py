import gc
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import ADAPTER_NAME, PAPER, TrainSettings
from .data import subset
from .features import decode_spans, prepare_features
from .io import environment, new_run_dir, write_json
from .metrics import score
from .model import adapter_parameter, freeze_for_training, head_parameter, load_model, parameter_digest


def predict(model, tokenizer, rows, decode, batch_size, device):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    features, feature_report = prepare_features(rows, tokenizer, decode)
    starts, ends = [], []
    model.eval()
    with torch.inference_mode():
        for index in range(0, len(features), batch_size):
            batch = features[index:index + batch_size]
            inputs = {key: torch.tensor([f["inputs"][key] for f in batch], device=device)
                      for key in batch[0]["inputs"]}
            outputs = model(**inputs)
            if outputs.start_logits.shape != inputs["input_ids"].shape or outputs.end_logits.shape != inputs["input_ids"].shape:
                raise ValueError("QA head did not produce two token-level logits")
            starts.extend(outputs.start_logits.detach().cpu().float().numpy())
            ends.extend(outputs.end_logits.detach().cpu().float().numpy())
            if index and (index // batch_size) % 25 == 0:
                print(f"Inference: {min(index + batch_size, len(features))}/{len(features)} windows", flush=True)
    predictions, spans = decode_spans(rows, features, starts, ends, decode)
    for row in rows:
        span = spans[row["id"]]
        if row["context"][span["start"]:span["end"]] != span["text"]:
            raise AssertionError("Prediction was not extracted from the original context")
    return predictions, spans, feature_report, starts, ends


def evaluate_model(model, tokenizer, rows, decode, batch_size, device, output_dir,
                   label, split, model_metadata, data_report, training=None):
    output_dir = new_run_dir(output_dir)
    started = time.monotonic()
    predictions, spans, features, starts, ends = predict(model, tokenizer, rows, decode, batch_size, device)
    metrics, details = score(rows, predictions)
    metrics.update({"label": label, "split": split,
                    "full_published_split": len(rows) == data_report["splits"][split]["rows"]})
    write_json(output_dir / "predictions.json", predictions)
    row_map = {row["id"]: row for row in rows}
    with (output_dir / "per-question.jsonl").open("w") as stream:
        for item in details:
            row = row_map[item["id"]]
            stream.write(json.dumps({**item, "question": row["question"], "span": spans[item["id"]]}, ensure_ascii=False) + "\n")
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run.json", {"environment": environment(), "model": model_metadata,
               "data": data_report, "decode": asdict(decode), "decode_settings_origin": "implementation choices; original settings unavailable",
               "training": training, "features": features, "eval_batch_size": batch_size,
               "elapsed_seconds": time.monotonic() - started, "paper_published_reference": PAPER,
               "claim": "Official benchmark evaluation; substantial train/test context overlap; paper scores are reference values, not verified reproduction."})
    print(json.dumps(metrics), flush=True)
    return {"metrics": metrics, "predictions": predictions, "start_logits": starts, "end_logits": ends}


def _collate(features):
    return {key: torch.tensor([f[key] for f in features], dtype=torch.long) for key in features[0]}


def save_checkpoint(model, tokenizer, path, record, optimizer=None, scheduler=None, step=None, epoch=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_adapter(str(path), ADAPTER_NAME, with_head=True, use_safetensors=True)
    tokenizer.save_pretrained(path)
    write_json(path / "training_metadata.json", record)
    write_json(path / "training_state.json", {"global_step": step, "epoch": epoch})
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "global_step": step, "epoch": epoch}, path / "optimizer.pt")
    if not (path / "head_config.json").exists():
        raise AssertionError("QA head was not saved with the adapter")


def train_model(model, tokenizer, splits, decode, settings, device, output_dir, model_metadata, data_report):
    settings.validate()
    output_dir = new_run_dir(output_dir)
    train_rows, validation = subset(splits["train"], settings.train_limit), subset(splits["validation"], settings.eval_limit)
    features, feature_report = prepare_features(train_rows, tokenizer, decode, training=True,
                                               invalid_policy=settings.invalid_train_answers)
    trainability = freeze_for_training(model)
    frozen_before = parameter_digest(model, lambda n, p: not p.requires_grad)
    adapter_before = parameter_digest(model, lambda n, p: adapter_parameter(n))
    head_before = parameter_digest(model, lambda n, p: head_parameter(n))
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader([f["inputs"] for f in features], batch_size=settings.batch_size,
                        shuffle=True, collate_fn=_collate, generator=generator, num_workers=0)
    updates_per_epoch = math.ceil(len(loader) / settings.gradient_accumulation_steps)
    total_steps = settings.max_steps or settings.epochs * updates_per_epoch
    epochs_to_run = math.ceil(total_steps / updates_per_epoch) if settings.max_steps else settings.epochs
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if name.endswith("bias") or parameter.ndim == 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": settings.weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=settings.learning_rate, betas=(settings.adam_beta1, settings.adam_beta2), eps=settings.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, settings.warmup_steps, total_steps)
    record = {"environment": environment(), "model": model_metadata, "data": data_report,
              "training": asdict(settings), "decode": asdict(decode), "features": feature_report,
              "trainability": trainability, "total_planned_steps": total_steps,
              "selection_rule": "last completed checkpoint" if settings.selection == "last" else "highest validation token F1; earliest checkpoint wins ties",
              "optimizer_groups": "weight decay excludes biases and one-dimensional normalization parameters",
              "training_label_policy": "first valid reference; CLS for windows without that answer",
              "precision": "float32", "evaluation_cadence": "end of each epoch (including a truncated last epoch)",
              "resume_supported": False}
    write_json(output_dir / "training_metadata.json", record)
    history, step, best_f1, best_path = [], 0, -1.0, None
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs_to_run):
        model.train()
        loss_sum, microbatches = 0.0, 0
        for batch_index, batch in enumerate(loader):
            group_start = (batch_index // settings.gradient_accumulation_steps) * settings.gradient_accumulation_steps
            group_size = min(settings.gradient_accumulation_steps, len(loader) - group_start)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss_sum += float(loss.detach().cpu())
            microbatches += 1
            (loss / group_size).backward()
            if (batch_index + 1) % settings.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], settings.max_grad_norm,
                                               error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress_epoch = epoch + (batch_index + 1) / len(loader)
                if step <= 2 or step % 25 == 0:
                    print(f"Train: step={step}/{total_steps}, epoch={progress_epoch:.3f}, loss={float(loss.detach().cpu()):.4f}", flush=True)
                if step % settings.save_steps == 0:
                    save_checkpoint(model, tokenizer, output_dir / f"checkpoint-{step}", record, optimizer, scheduler, step, progress_epoch)
                if step >= total_steps:
                    break
        result = evaluate_model(model, tokenizer, validation, decode, settings.eval_batch_size, device,
                                output_dir / f"validation-epoch-{epoch + 1}", "AgRoBERTa reimplementation", "validation",
                                model_metadata, data_report, training=asdict(settings))
        history.append({"epoch": progress_epoch, "global_step": step, "mean_training_loss": loss_sum / microbatches,
                        "validation": result["metrics"]})
        write_json(output_dir / "history.json", history)
        if settings.selection == "best-validation-f1" and result["metrics"]["f1"] > best_f1:
            best_f1 = result["metrics"]["f1"]
            best_path = output_dir / f"best-step-{step}"
            save_checkpoint(model, tokenizer, best_path, record, optimizer, scheduler, step, progress_epoch)
        if step >= total_steps:
            break
    frozen_after = parameter_digest(model, lambda n, p: not p.requires_grad)
    proof = {"backbone_frozen_unchanged": frozen_before == frozen_after,
             "frozen_sha256_before": frozen_before, "frozen_sha256_after": frozen_after,
             "adapter_updated": adapter_before != parameter_digest(model, lambda n, p: adapter_parameter(n)),
             "qa_head_updated": head_before != parameter_digest(model, lambda n, p: head_parameter(n))}
    if not all(proof[k] for k in ("backbone_frozen_unchanged", "adapter_updated", "qa_head_updated")):
        raise AssertionError(f"Training verification failed: {proof}")
    selected = best_path if settings.selection == "best-validation-f1" else output_dir / "final"
    record.update({"completed_steps": step, "completed_epochs": progress_epoch, "verification": proof,
                   "elapsed_seconds": time.monotonic() - started, "selected_checkpoint": str(selected.resolve()),
                   "full_20_epoch_published_run": settings == TrainSettings()})
    save_checkpoint(model, tokenizer, output_dir / "final", record, optimizer, scheduler, step, progress_epoch)
    # Include final verification and the selection rule in the selected checkpoint's record.
    if best_path:
        write_json(best_path / "training_metadata.json", record)
    write_json(output_dir / "training_metadata.json", record)
    return {"checkpoint": str(selected), "record": record, "last_validation": result}


def run_smoke(load_kwargs, splits, decode, settings, device, output_dir, data_report):
    settings.validate()
    if settings.selection != "last":
        raise ValueError("Smoke verification requires last-checkpoint selection")
    output_dir = new_run_dir(output_dir)
    model, tokenizer, metadata = load_model(**load_kwargs)
    rows = subset(splits["validation"], settings.eval_limit)
    evaluate_model(model, tokenizer, rows, decode, settings.eval_batch_size, device,
                   output_dir / "before-validation", "before fine-tuning", "validation", metadata, data_report)
    trained = train_model(model, tokenizer, splits, decode, settings, device, output_dir / "training", metadata, data_report)
    expected_hash = parameter_digest(model, lambda n, p: adapter_parameter(n) or head_parameter(n))
    expected = trained["last_validation"]
    model.to("cpu")
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    reload_kwargs = {**load_kwargs, "adapter": trained["checkpoint"], "head": None,
                     "tokenizer_path": trained["checkpoint"]}
    reloaded, tokenizer, metadata = load_model(**reload_kwargs)
    actual_hash = parameter_digest(reloaded, lambda n, p: adapter_parameter(n) or head_parameter(n))
    actual = evaluate_model(reloaded, tokenizer, rows, decode, settings.eval_batch_size, device,
                            output_dir / "reloaded-validation", "AgRoBERTa reimplementation (smoke only)",
                            "validation", metadata, data_report, training=trained["record"])
    max_diff = max(float(np.max(np.abs(np.asarray(expected[key]) - np.asarray(actual[key]))))
                   for key in ("start_logits", "end_logits"))
    report = {"status": "passed", "training": trained["record"]["verification"],
              "adapter_and_head_reload_identical": expected_hash == actual_hash,
              "predictions_reload_identical": expected["predictions"] == actual["predictions"],
              "reload_max_absolute_logit_difference": max_diff, "checkpoint": trained["checkpoint"],
              "scope": "Short integration test; not a 20-epoch reproduction"}
    if not report["adapter_and_head_reload_identical"] or not report["predictions_reload_identical"] or max_diff > 1e-5:
        raise AssertionError(f"Reload verification failed: {report}")
    write_json(output_dir / "smoke-report.json", report)
    return report
