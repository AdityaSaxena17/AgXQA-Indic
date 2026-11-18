from dataclasses import asdict, dataclass

DATASET = "msu-ceco/agxqa_v1"
DATA_REVISION = "9fb0cf2a6d03ccfccbd67d5966179b43b5b77e2c"
BACKBONE = "msu-ceco/roberta-ft-on-agextcorpus-2023-12-10_v2"
BACKBONE_REVISION = "bdfdd689aede30b4d7d84cb6e267b98a88d58e40"
ADAPTER = "AdapterHub/roberta-base_qa_squad1_pfeiffer"
ADAPTER_REVISION = "059c341c1e82b0b8e418d76a2c5113af41326310"
AUTHORS_RELEASE = "msu-ceco/agroberta_2024-01-11_21-02-21"
AUTHORS_REVISION = "b76fdef39422cd4188aedc2d3c79328913117545"
COUNTS = {"train": 1503, "validation": 353, "test": 330}
FILES = {split: f"agxqa-{split}-2024-06-11.jsonl" for split in COUNTS}
# SHA-256 of the exact files downloaded from DATA_REVISION.
DATA_SHA256 = {
    "train": "72da2aaa34d90ce32f28c6d461f258dc9ec712f0af8ff3b5041ceb1528817d85",
    "validation": "386181aa34e6b71faa3474868008cc7b8bed7143da0218f22e546cdfc4ebc583",
    "test": "505fa18e7d7e7b6f86f090ae8c1172d5fd29ba5f6845a82107eb49df59c32a8b",
}
PAPER = {"exact_match": 55.15, "f1": 78.89}
ADAPTER_NAME = "agxqa"


@dataclass
class DecodeSettings:
    # Implementation choices: these were not established by the released args.
    max_length: int = 384
    stride: int = 128
    n_best: int = 20
    max_answer_length: int = 30
    allow_no_answer: bool = False
    null_score_diff_threshold: float = 0.0

    def validate(self):
        if self.max_length < 16 or not 0 <= self.stride < self.max_length:
            raise ValueError("Require max_length >= 16 and 0 <= stride < max_length")
        if self.n_best < 1 or self.max_answer_length < 1:
            raise ValueError("Candidate and answer limits must be positive")


@dataclass
class TrainSettings:
    epochs: int = 20
    learning_rate: float = 5e-5
    batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    seed: int = 42
    save_steps: int = 500
    max_steps: int | None = None
    train_limit: int | None = None
    eval_limit: int | None = None
    selection: str = "last"
    invalid_train_answers: str = "skip"
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    fp16: bool = False
    bf16: bool = False
    load_best_model_at_end: bool = False

    def validate(self):
        for field in ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps", "save_steps"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        for field in ("max_steps", "train_limit", "eval_limit"):
            if getattr(self, field) is not None and getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive when specified")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.warmup_steps < 0:
            raise ValueError("Invalid optimizer settings")


def settings_dict(value):
    return asdict(value)
