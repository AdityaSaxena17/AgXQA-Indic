import hashlib
import json
from pathlib import Path

import torch
from adapters import AutoAdapterModel
from huggingface_hub import list_repo_files, snapshot_download
from transformers import AutoTokenizer, set_seed

from .config import ADAPTER, ADAPTER_NAME, ADAPTER_REVISION, BACKBONE, BACKBONE_REVISION
from .io import sha256, write_json


def download_models(directory):
    directory = Path(directory)
    result = {}
    for name, repo, revision in (("backbone", BACKBONE, BACKBONE_REVISION), ("squad", ADAPTER, ADAPTER_REVISION)):
        files = list_repo_files(repo, revision=revision)
        wanted = {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                  "special_tokens_map.json", "added_tokens.json", "adapter_config.json", "head_config.json",
                  "pytorch_adapter.bin", "pytorch_model_head.bin"}
        wanted.add("model.safetensors" if "model.safetensors" in files else "pytorch_model.bin")
        path = Path(snapshot_download(repo, revision=revision, local_dir=directory / name,
                                      allow_patterns=sorted(wanted), max_workers=2))
        result[name] = {"repo": repo, "revision": revision,
                        "files": {p.name: sha256(p) for p in path.iterdir()
                                  if p.is_file() and p.name != "source-manifest.json"}}
        write_json(path / "source-manifest.json", result[name])
    write_json(directory / "download-manifest.json", result)
    return result


def configure_runtime(device="auto", threads=4, seed=42):
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    set_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def source_metadata(source, revision):
    path = Path(source)
    if not path.is_dir():
        return {"repo": source, "revision": revision}
    manifest = path / "source-manifest.json"
    return {"local_path": str(path.resolve()),
            "download_source": json.loads(manifest.read_text()) if manifest.exists() else None,
            "artifact_hashes": {p.name: sha256(p) for p in path.iterdir() if p.is_file()
                                and p.suffix in (".json", ".bin", ".safetensors")}}


def load_model(base=BACKBONE, base_revision=BACKBONE_REVISION, adapter=ADAPTER,
               adapter_revision=ADAPTER_REVISION, head=None, tokenizer_path=None,
               tokenizer_revision=None, device="cpu"):
    requested_adapter = {"source": adapter, "revision": adapter_revision if not Path(adapter).is_dir() else None}
    base_local = Path(base).is_dir()
    # Eager attention is an explicit implementation choice, portable across CPU/MPS/CUDA.
    model, loading_info = AutoAdapterModel.from_pretrained(
        base, revision=None if base_local else base_revision,
        attn_implementation="eager", output_loading_info=True)
    missing_backbone = [k for k in loading_info["missing_keys"]
                        if "heads." not in k and "pooler" not in k and "adapters." not in k]
    if missing_backbone or loading_info.get("mismatched_keys"):
        raise ValueError(f"Backbone weights were not fully loaded: {loading_info}")
    if not Path(adapter).is_dir():
        adapter = snapshot_download(adapter, revision=adapter_revision,
                                    allow_patterns=["adapter_config.json", "head_config.json", "pytorch_adapter.bin",
                                                    "pytorch_model_head.bin", "adapter.safetensors", "model_head.safetensors"])
    adapter_path = Path(adapter)
    if not (adapter_path / "adapter_config.json").exists():
        raise ValueError("Adapter config is missing; the authors' public tokenizer-only release is not a QA checkpoint")
    if not any((adapter_path / name).exists() for name in ("pytorch_adapter.bin", "adapter.safetensors")):
        raise ValueError("Adapter weights are missing")
    head_path = Path(head) if head else adapter_path
    if not (head_path / "head_config.json").exists() or not any(
        (head_path / name).exists() for name in ("pytorch_model_head.bin", "model_head.safetensors")):
        raise ValueError("A trained QA head config and weights are required")
    head_config = json.loads((head_path / "head_config.json").read_text())
    actual_config = head_config.get("config", head_config)
    if actual_config.get("head_type") != "question_answering" or actual_config.get("num_labels") != 2:
        raise ValueError(f"Expected an existing two-logit QA head, got {head_config}")
    adapter_info = {}
    model.load_adapter(str(adapter_path), load_as=ADAPTER_NAME, with_head=head is None,
                       set_active=True, loading_info=adapter_info, use_safetensors=True)
    if head:
        model.load_head(str(head_path), load_as=ADAPTER_NAME, loading_info=adapter_info, use_safetensors=True)
    if adapter_info.get("missing_keys") or adapter_info.get("unexpected_keys"):
        raise ValueError(f"Adapter/head weights failed to load exactly: {adapter_info}")
    model.set_active_adapters(ADAPTER_NAME)
    model.active_head = ADAPTER_NAME
    # The MLM backbone may also have imported an MLM head; remove unused heads.
    for name in list(model.heads):
        if name != ADAPTER_NAME:
            model.delete_head(name)
    if (ADAPTER_NAME not in model.heads or model.active_head != ADAPTER_NAME
            or model.active_adapters.flatten() != {ADAPTER_NAME}):
        raise ValueError("QA head was not loaded and activated")
    tokenizer_source = tokenizer_path or base
    tokenizer_local = Path(tokenizer_source).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True,
                    revision=None if tokenizer_local else (tokenizer_revision or base_revision))
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for context offsets")
    model.to(device)
    metadata = {"backbone": source_metadata(base, base_revision),
                "requested_adapter": requested_adapter,
                "adapter": source_metadata(str(adapter_path), adapter_revision),
                "head": source_metadata(str(head_path), adapter_revision),
                "tokenizer": source_metadata(tokenizer_source, tokenizer_revision or base_revision),
                "backbone_load_info": loading_info, "adapter_load_info": adapter_info,
                "active_adapters": str(model.active_adapters), "active_head": model.active_head,
                "attention_implementation": "eager", "device": str(device)}
    return model, tokenizer, metadata


def parameter_digest(model, predicate):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if predicate(name, parameter):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def adapter_parameter(name):
    return f".adapters.{ADAPTER_NAME}." in name


def head_parameter(name):
    return name.startswith(f"heads.{ADAPTER_NAME}.")


def freeze_for_training(model):
    model.train_adapter(ADAPTER_NAME)
    model.active_head = ADAPTER_NAME
    trainable = []
    for name, parameter in model.named_parameters():
        expected = adapter_parameter(name) or head_parameter(name)
        if parameter.requires_grad != expected:
            raise ValueError(f"Unexpected trainability: {name}, requires_grad={parameter.requires_grad}")
        if expected:
            trainable.append(name)
    if not any(adapter_parameter(n) for n in trainable) or not any(head_parameter(n) for n in trainable):
        raise ValueError("Both adapter and QA head must be trainable")
    return {"trainable_names": trainable, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
