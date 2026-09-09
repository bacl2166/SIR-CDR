from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetConfig:
    pairs: tuple[tuple[str, str, str], ...] = (
        ("amazon", "Sports", "Clothing"),
        ("amazon", "Clothing", "Sports"),
        ("amazon", "Phones", "Electronics"),
        ("amazon", "Electronics", "Phones"),
        ("douban", "Books", "Movies"),
        ("douban", "Movies", "Books"),
    )
    min_user_interactions: int = 5
    validation_holdout: int = 1
    test_holdout: int = 1

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("DatasetConfig.pairs must contain at least one cross-domain pair.")
        if self.min_user_interactions < self.validation_holdout + self.test_holdout + 1:
            raise ValueError("min_user_interactions must leave room for train, validation, and test events.")


@dataclass(frozen=True)
class EmbeddingConfig:
    enable_api_calls: bool = False
    provider_order: tuple[str, ...] = ("qwen", "deepseek", "local", "other")
    cache_dir: str = "artifacts/embeddings"
    text_dim: int = 768
    image_dim: int = 768

    def __post_init__(self) -> None:
        if self.text_dim <= 0 or self.image_dim <= 0:
            raise ValueError("Embedding dimensions must be positive.")


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    codebook_size: int = 512
    token_length: int = 4
    prefix_length: int = 4
    reasoning_steps: int = 3
    graph_layers: int = 2

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "codebook_size", "token_length", "prefix_length", "reasoning_steps", "graph_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"ModelConfig.{name} must be positive.")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    warmup_epochs: int = 5

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("TrainingConfig batch_size and max_epochs must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("TrainingConfig learning_rate must be positive and weight_decay non-negative.")


@dataclass(frozen=True)
class FrameworkConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
