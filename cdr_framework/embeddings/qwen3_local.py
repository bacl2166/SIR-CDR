from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from typing import Any

import torch
import torch.nn.functional as F

from cdr_framework.datasets.schema import ItemMetadataRecord
from cdr_framework.embeddings.providers import BaseTextEmbeddingProvider


class Qwen3EmbeddingProvider(BaseTextEmbeddingProvider):
    """Lazy local inference with the official Qwen3-Embedding-8B weights."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        dim: int = 768,
        batch_size: int = 1,
        device: str = "cuda",
        max_sequence_length: int = 8192,
        model_path_env: str = "QWEN3_EMBEDDING_MODEL_PATH",
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        if dim <= 0 or dim > 4096:
            raise ValueError("Qwen3 embedding dimension must be in [1, 4096].")
        if batch_size <= 0 or max_sequence_length <= 0:
            raise ValueError("batch_size and max_sequence_length must be positive.")
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.model_path_env = model_path_env
        self._model_loader = model_loader
        self._encoder: Any | None = None

    def _get_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        if self._model_loader is None:
            from sentence_transformers import SentenceTransformer

            self._model_loader = SentenceTransformer
        model_name_or_path = os.environ.get(self.model_path_env) or self.model
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._encoder = self._model_loader(
            model_name_or_path,
            device=self.device,
            model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
        )
        if hasattr(self._encoder, "tokenizer"):
            self._encoder.tokenizer.padding_side = "left"
        self._encoder.max_seq_length = self.max_sequence_length
        return self._encoder

    def encode_text(self, records: Sequence[str | ItemMetadataRecord]) -> torch.Tensor:
        texts = [_embedding_text(record) for record in records]
        if not texts:
            return torch.empty((0, self.dim), dtype=torch.float32)
        vectors = self._get_encoder().encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vectors = torch.as_tensor(vectors).detach().to(device="cpu", dtype=torch.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] < self.dim:
            raise RuntimeError(
                f"Qwen3-Embedding-8B returned shape {tuple(vectors.shape)}, "
                f"which cannot provide {(len(texts), self.dim)}."
            )
        vectors = F.normalize(vectors[:, : self.dim], p=2, dim=1)
        if not torch.isfinite(vectors).all():
            raise RuntimeError("Qwen3-Embedding-8B returned non-finite vectors.")
        return vectors


def _embedding_text(record: str | ItemMetadataRecord) -> str:
    if isinstance(record, str):
        return record
    fields = [record.title, record.brand_or_creator, *record.category, record.description]
    return " ".join(str(value).strip() for value in fields if value).strip()
