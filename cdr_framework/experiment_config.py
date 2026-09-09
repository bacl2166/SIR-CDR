from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AmazonPreprocessingConfig:
    source_domain: str = "Sports_and_Outdoors"
    target_domain: str = "Clothing_Shoes_and_Jewelry"
    raw_dir: Path = Path("data/raw/amazon2014")
    processed_dir: Path = Path("data/processed/sports_to_clothing")
    min_interactions_per_domain: int = 5
    max_text_chars: int = 12000
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_dir", Path(self.raw_dir))
        object.__setattr__(self, "processed_dir", Path(self.processed_dir))

        if self.source_domain != "Sports_and_Outdoors":
            raise ValueError("source_domain must be Sports_and_Outdoors.")
        if self.target_domain != "Clothing_Shoes_and_Jewelry":
            raise ValueError("target_domain must be Clothing_Shoes_and_Jewelry.")
        if self.min_interactions_per_domain < 3:
            raise ValueError("min_interactions_per_domain must be at least 3.")
        if self.max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AmazonPreprocessingConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        dataset = payload.get("dataset", {})
        return cls(**dataset)


@dataclass(frozen=True)
class QwenEmbeddingConfig:
    input_path: Path = Path("data/processed/sports_to_clothing/item_texts.jsonl")
    output_dir: Path = Path("artifacts/embeddings/sports_to_clothing")
    provider: str = "qwen"
    model: str = "text-embedding-v4"
    dimension: int = 768
    batch_size: int = 10
    max_retries: int = 5
    retry_initial_seconds: float = 2.0
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url_env: str = "DASHSCOPE_BASE_URL"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.provider.lower() != "qwen":
            raise ValueError("The first formal embedding run requires provider=qwen.")
        if not self.model:
            raise ValueError("Embedding model must not be empty.")
        if self.dimension <= 0 or self.batch_size <= 0:
            raise ValueError("Embedding dimension and batch_size must be positive.")
        if self.max_retries < 0 or self.retry_initial_seconds < 0:
            raise ValueError("Retry settings must be non-negative.")
        if not self.api_key_env or not self.base_url_env:
            raise ValueError("Embedding credential environment names must not be empty.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QwenEmbeddingConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**payload.get("embedding", {}))


@dataclass(frozen=True)
class TokenizerTrainingConfig:
    embeddings_path: Path = Path("artifacts/embeddings/sports_to_clothing/text_embeddings.pt")
    item_texts_path: Path = Path("data/processed/sports_to_clothing/item_texts.jsonl")
    output_dir: Path = Path("artifacts/tokenizer/sports_to_clothing")
    input_dim: int = 768
    hidden_dim: int = 128
    codebook_size: int = 512
    token_length: int = 4
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    domain_loss_weight: float = 0.1
    gate_balance_weight: float = 0.01
    kmeans_iterations: int = 20
    max_collision_rate: float = 0.10
    min_level_utilization: float = 0.10
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("embeddings_path", "item_texts_path", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "input_dim",
            "hidden_dim",
            "codebook_size",
            "token_length",
            "batch_size",
            "max_epochs",
            "kmeans_iterations",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Tokenizer {name} must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Tokenizer optimizer settings are invalid.")
        if self.domain_loss_weight < 0 or self.gate_balance_weight < 0:
            raise ValueError("Tokenizer loss weights must be non-negative.")
        if not 0 <= self.max_collision_rate < 1:
            raise ValueError("Tokenizer max_collision_rate must be in [0, 1).")
        if not 0 < self.min_level_utilization <= 1:
            raise ValueError("Tokenizer min_level_utilization must be in (0, 1].")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TokenizerTrainingConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**payload.get("tokenizer", {}))


@dataclass(frozen=True)
class RecommendationTrainingConfig:
    processed_dir: Path = Path("data/processed/sports_to_clothing")
    tokenizer_dir: Path = Path("artifacts/tokenizer/sports_to_clothing")
    output_dir: Path = Path("artifacts/recommendation/sports_to_clothing")
    hidden_dim: int = 128
    reasoning_steps: int = 3
    prefix_length: int = 4
    max_sequence_length: int = 50
    batch_size: int = 256
    evaluation_batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    evaluation_every: int = 5
    patience: int = 5
    retrieval_temperature: float = 0.1
    top_ks: tuple[int, ...] = (5, 10, 20)
    candidate_chunk_size: int = 2048
    rerank_candidates: int = 200
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("processed_dir", "tokenizer_dir", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(self, "top_ks", tuple(int(k) for k in self.top_ks))
        for name in (
            "hidden_dim",
            "reasoning_steps",
            "prefix_length",
            "max_sequence_length",
            "batch_size",
            "evaluation_batch_size",
            "max_epochs",
            "evaluation_every",
            "patience",
            "candidate_chunk_size",
            "rerank_candidates",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Recommendation {name} must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Recommendation optimizer settings are invalid.")
        if self.retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be positive.")
        if not self.top_ks or any(k <= 0 for k in self.top_ks):
            raise ValueError("top_ks must contain positive cutoffs.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RecommendationTrainingConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**payload.get("recommendation", {}))
