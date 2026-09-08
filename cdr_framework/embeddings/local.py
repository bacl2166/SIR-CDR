from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from cdr_framework.datasets.schema import ItemMetadataRecord
from cdr_framework.embeddings.providers import BaseTextEmbeddingProvider


class DeterministicHashTextEmbeddingProvider(BaseTextEmbeddingProvider):
    def __init__(self, dim: int = 768):
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.dim = dim

    def encode_text(self, records: Sequence[str | ItemMetadataRecord]) -> torch.Tensor:
        vectors = [_hash_to_vector(_record_text(record), self.dim) for record in records]
        if not vectors:
            return torch.empty((0, self.dim), dtype=torch.float32)
        return F.normalize(torch.stack(vectors), dim=-1)


def _record_text(record: str | ItemMetadataRecord) -> str:
    if isinstance(record, str):
        return record
    pieces = [
        record.title,
        record.domain,
        " ".join(record.category),
        record.brand_or_creator or "",
        record.description or "",
        " ".join(f"{key}:{value}" for key, value in sorted(record.attributes.items())),
    ]
    return " | ".join(piece for piece in pieces if piece)


def _hash_to_vector(text: str, dim: int) -> torch.Tensor:
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    return torch.tensor(values[:dim], dtype=torch.float32)
