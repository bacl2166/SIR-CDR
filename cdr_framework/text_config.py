from dataclasses import dataclass
from pathlib import Path
import math
import yaml

from cdr_framework.experiment_config import RecommendationTrainingConfig


@dataclass(frozen=True)
class TextCDRConfig(RecommendationTrainingConfig):
    output_dir: Path = Path("artifacts/text_cdr/sports_to_clothing/full_seed42")
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    evaluation_every: int = 2
    evaluation_batch_size: int = 16
    patience: int = 8
    dropout: float = 0.1
    graph_layers: int = 2
    cross_window: int = 5
    feedback_steps: int = 2
    graph_enabled: bool = True
    reasoning_enabled: bool = True
    semantic_enabled: bool = True
    cpf_weight: float = 0.1
    alignment_weight: float = 0.02
    separation_weight: float = 0.01
    generation_weight: float = 1.0
    retrieval_weight: float = 1.0
    label_smoothing: float = 0.0  # generation CE label smoothing (de-biases overconfidence / popularity collapse)
    source_enabled: bool = True
    inference_mode: str = "hybrid"
    beam_size: int = 50
    decode_chunk_size: int = 256
    rerank_candidates: int = 500
    scheduler_patience: int = 2
    selection_metric: str = "NDCG@10"
    # Dual structural injection (new design). Defaults enable the full
    # architecture: prototype-side disentanglement, CD/SP structural
    # injectors, codebook-summary prefix, contrastive shared alignment
    # and source-private separation regularization.
    prototype_enabled: bool = True
    cd_injector_enabled: bool = True
    sp_injector_enabled: bool = True
    codebook_summary_enabled: bool = True
    contrastive_alignment: bool = True
    lsep_weight: float = 0.01
    proto_orth_weight: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        for name in ("graph_layers", "cross_window", "feedback_steps", "beam_size",
                     "decode_chunk_size", "scheduler_patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("cpf_weight", "alignment_weight", "separation_weight", "generation_weight",
                     "retrieval_weight", "lsep_weight", "proto_orth_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.generation_weight + self.retrieval_weight <= 0:
            raise ValueError("At least one ranking weight must be positive")
        if self.inference_mode not in {"retrieval", "hybrid", "exhaustive", "generate"}:
            raise ValueError("Unknown inference_mode")
        if self.selection_metric not in {f"{metric}@{k}" for metric in ("HR", "NDCG", "MRR") for k in self.top_ks}:
            raise ValueError("selection_metric must match top_ks")

    @classmethod
    def from_yaml(cls, path):
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "text_recommendation" not in payload:
            raise ValueError("Expected text_recommendation section, not legacy recommendation")
        return cls(**payload["text_recommendation"])
