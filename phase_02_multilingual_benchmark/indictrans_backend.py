"""IndicTrans2 backend. Neural loading happens only when Backend is instantiated."""
import importlib.metadata
import json
from pathlib import Path
import re

from .pipeline import canonical_text, file_digest


def chunk_text(text, budget, measure):
    """Keep all normalized source words; never silently truncate long inputs."""
    text = canonical_text(text)
    if not text:
        raise ValueError("Empty translation input")
    if measure(text) <= budget:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    chunks, current = [], ""
    for sentence in sentences:
        combined = (current + " " + sentence).strip()
        if measure(combined) <= budget:
            current = combined
            continue
        if current:
            chunks.append(current)
            current = ""
        if measure(sentence) <= budget:
            current = sentence
            continue
        words = sentence.split()
        while words:
            low, high, take = 1, len(words), 0
            while low <= high:
                middle = (low + high) // 2
                if measure(" ".join(words[:middle])) <= budget:
                    take = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if take == 0:
                raise ValueError("A source word exceeds the input-token budget; manual segmentation required")
            chunks.append(" ".join(words[:take]))
            words = words[take:]
    if current:
        chunks.append(current)
    if " ".join(chunks) != text or any(measure(c) > budget for c in chunks):
        raise ValueError("Chunking lost source text or exceeded the token budget")
    return chunks


class Backend:
    def __init__(self, config, device="auto", local_model=None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError:
            from IndicTransToolkit.IndicTransToolkit import IndicProcessor
        if importlib.metadata.version("indictranstoolkit") != config["toolkit_version"]:
            raise ValueError("Install the pinned IndicTransToolkit version before translating")
        torch.set_num_threads(config["runtime"]["threads"])
        set_seed(config["seed"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.torch, self.config, self.Processor = torch, config, IndicProcessor
        source = str(local_model) if local_model else config["model_id"]
        revision = None if local_model else config["model_revision"]
        if local_model and not Path(local_model).is_dir():
            raise ValueError("--local-model must be a downloaded local checkpoint directory")
        if local_model:
            manifest = json.loads((Path(local_model)/"source-manifest.json").read_text())
            if manifest["repo"] != config["model_id"] or manifest["revision"] != config["model_revision"]:
                raise ValueError("Local checkpoint manifest does not match the selected model/revision")
            if not all(file_digest(Path(local_model)/name) == expected for name,expected in manifest["files"].items()):
                raise ValueError("Local checkpoint file hashes do not match its manifest")
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, code_revision=revision, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source, revision=revision, code_revision=revision,
                            trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager").to(self.device)
        self.model.eval()
        self.processor = IndicProcessor(inference=True)
        self.metadata = {"model_id":config["model_id"], "model_revision":config["model_revision"],
                         "device":str(self.device), "dtype":"float32", "attention":"eager", "trust_remote_code":True,
                         "local_model_supplied":bool(local_model),
                         "packages":{p:importlib.metadata.version(p) for p in ("torch","transformers","indictranstoolkit","huggingface-hub","sentencepiece")}}

    def measure(self, text, target):
        # Keep length-probe entity mappings separate from the real batch processor.
        probe = self.Processor(inference=True)
        prepared = probe.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=target)
        return len(self.tokenizer(prepared, truncation=False, padding=False)["input_ids"][0])

    def translate(self, text, target):
        cfg = self.config
        chunks = chunk_text(text, cfg["generation"]["max_source_tokens"], lambda value:self.measure(value,target))
        records = []
        batch_size = cfg["runtime"]["batch_size"]
        for start in range(0,len(chunks),batch_size):
            batch = chunks[start:start+batch_size]
            prepared = self.processor.preprocess_batch(batch,src_lang="eng_Latn",tgt_lang=target)
            inputs = self.tokenizer(prepared,truncation=False,padding="longest",return_tensors="pt",return_attention_mask=True).to(self.device)
            lengths = inputs["attention_mask"].sum(dim=1).tolist()
            if any(length > cfg["generation"]["max_source_tokens"] for length in lengths):
                raise ValueError("Prepared input exceeded the source budget; inference was not truncated")
            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs,do_sample=False,num_beams=cfg["generation"]["num_beams"],
                             num_return_sequences=1,min_length=0,max_length=cfg["generation"]["max_target_tokens"],use_cache=True)
            token_rows = outputs.detach().cpu().tolist()
            decoded = self.tokenizer.batch_decode(outputs,skip_special_tokens=True,clean_up_tokenization_spaces=True)
            translated = self.processor.postprocess_batch(decoded,lang=target)
            for source, prepared_text, decoded_text, translated_text, tokens, count in zip(batch,prepared,decoded,translated,token_rows,lengths):
                length = len([v for v in tokens if v != self.tokenizer.pad_token_id])
                records.append({"source_text":source,"prepared_source_text":prepared_text,"decoded_text":decoded_text,
                                "generated_token_ids":tokens,"translation_text":translated_text,"source_token_count":count,
                                "output_token_count":length,"target_length_limit_reached":length >= cfg["generation"]["max_target_tokens"]})
        return {"text":" ".join(row["translation_text"] for row in records),"segments":records,
                "generation_issues":["target_generation_length_limit_reached"] if any(r["target_length_limit_reached"] for r in records) else []}
