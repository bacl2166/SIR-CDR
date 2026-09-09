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
