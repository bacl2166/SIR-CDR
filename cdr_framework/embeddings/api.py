from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from cdr_framework.config import EmbeddingConfig
from cdr_framework.datasets.schema import ItemMetadataRecord
from cdr_framework.embeddings.local import DeterministicHashTextEmbeddingProvider
from cdr_framework.embeddings.providers import BaseTextEmbeddingProvider


@dataclass(frozen=True)
class ReservedAPITextEmbeddingProvider(BaseTextEmbeddingProvider):
    provider_name: str
    api_key_env: str
    model: str
    dim: int = 768

    def encode_text(self, records: Sequence[str | ItemMetadataRecord]) -> torch.Tensor:
        del records
        raise RuntimeError(
            f"{self.provider_name} text embedding API is reserved but not implemented. "
            f"Set up the provider in cdr_framework.embeddings.api, read credentials from {self.api_key_env}, "
            "and keep cache-first execution enabled before using external calls."
        )


class QwenTextEmbeddingProvider(ReservedAPITextEmbeddingProvider):
    def __init__(self, api_key_env: str = "DASHSCOPE_API_KEY", model: str = "text-embedding-v4", dim: int = 768):
        super().__init__(provider_name="Qwen", api_key_env=api_key_env, model=model, dim=dim)


class DeepSeekTextEmbeddingProvider(ReservedAPITextEmbeddingProvider):
    def __init__(self, api_key_env: str = "DEEPSEEK_API_KEY", model: str = "deepseek-embedding", dim: int = 768):
        super().__init__(provider_name="DeepSeek", api_key_env=api_key_env, model=model, dim=dim)


def build_text_embedding_provider(config: EmbeddingConfig) -> BaseTextEmbeddingProvider:
    if not config.enable_api_calls:
        return DeterministicHashTextEmbeddingProvider(dim=config.text_dim)

    for provider in config.provider_order:
        normalized = provider.lower()
        if normalized == "qwen":
            return QwenTextEmbeddingProvider(dim=config.text_dim)
        if normalized == "deepseek":
            return DeepSeekTextEmbeddingProvider(dim=config.text_dim)
        if normalized == "local":
            return DeterministicHashTextEmbeddingProvider(dim=config.text_dim)
    raise ValueError(f"No supported text embedding provider in order: {config.provider_order!r}")
