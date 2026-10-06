"""Opt-in LLM experiment settings; legacy TextCDRConfig is unchanged."""
from dataclasses import dataclass
from pathlib import Path
import math
import re
import yaml

from cdr_framework.text_config import TextCDRConfig


@dataclass(frozen=True)
class LLMCDRConfig(TextCDRConfig):
    output_dir: Path = Path("artifacts/llm_cdr_v1/sports_to_clothing/seed42")
    llm_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    llm_revision: str = "main"  # prepare resolves this to an immutable commit
    llm_dtype: str = "bfloat16"
    llm_local_files_only: bool = True
    llm_gradient_checkpointing: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    reasoning_steps: int = 2
    feedback_steps: int = 1
    batch_size: int = 1
    evaluation_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 0.0001
    contrastive_alignment: bool = False  # microbatch=1 has no in-batch negatives

    def __post_init__(self):
        super().__post_init__()
        if not self.reasoning_enabled or self.feedback_steps != 1:
            raise ValueError("LLM v1 requires reasoning_enabled=true and feedback_steps=1")
        if self.batch_size == 1 and self.contrastive_alignment:
            raise ValueError("batch_size=1 requires cosine alignment; accumulation cannot add negatives")
        for name in ("reasoning_steps", "lora_rank", "lora_alpha", "gradient_accumulation_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.llm_dtype not in {"float32", "bfloat16"}:
            raise ValueError("llm_dtype must be float32 or bfloat16")
        if not math.isfinite(self.label_smoothing) or not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0,1)")
        if not math.isfinite(self.lora_dropout) or not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0,1)")
        if not self.llm_model_name or not self.llm_revision:
            raise ValueError("Model name and revision are required")

    def require_pinned_model(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.llm_revision):
            raise ValueError("Run --action prepare first: llm_revision must be a 40-character commit SHA")

    @classmethod
    def from_yaml(cls, path):
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "llm_recommendation" not in payload:
            raise ValueError("Expected llm_recommendation section")
        return cls(**payload["llm_recommendation"])
