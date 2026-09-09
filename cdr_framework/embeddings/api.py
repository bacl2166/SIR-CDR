from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import os
from typing import Any

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


class QwenTextEmbeddingProvider(BaseTextEmbeddingProvider):
    def __init__(
        self,
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url_env: str = "DASHSCOPE_BASE_URL",
        model: str = "text-embedding-v4",
        dim: int = 768,
        client: Any | None = None,
    ):
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.model = model
        self.dim = dim
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.api_key_env)
        base_url = os.environ.get(self.base_url_env)
        missing = [name for name, value in ((self.api_key_env, api_key), (self.base_url_env, base_url)) if not value]
        if missing:
            raise RuntimeError(
                "Missing required Qwen embedding environment variable(s): "
                + ", ".join(missing)
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def encode_text(self, records: Sequence[str | ItemMetadataRecord]) -> torch.Tensor:
        texts = [_embedding_text(record) for record in records]
        if not texts:
            return torch.empty((0, self.dim), dtype=torch.float32)
        response = self._get_client().embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dim,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda entry: getattr(entry, "index", 0))
        vectors = torch.tensor([entry.embedding for entry in ordered], dtype=torch.float32)
        if vectors.shape != (len(texts), self.dim):
            raise RuntimeError(
                f"Qwen returned embedding shape {tuple(vectors.shape)}, "
                f"expected {(len(texts), self.dim)}."
            )
        return vectors


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


def _embedding_text(record: str | ItemMetadataRecord) -> str:
    if isinstance(record, str):
        return record
    fields = [record.title, record.brand_or_creator, *record.category, record.description]
    return " ".join(str(value).strip() for value in fields if value).strip()
