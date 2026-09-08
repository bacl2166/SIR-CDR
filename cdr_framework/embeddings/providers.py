from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from cdr_framework.datasets.schema import ItemMetadataRecord


class BaseTextEmbeddingProvider(ABC):
    @abstractmethod
    def encode_text(self, records: Sequence[str | ItemMetadataRecord]) -> torch.Tensor:
        raise NotImplementedError


class BaseImageEmbeddingProvider(ABC):
    @abstractmethod
    def encode_image(self, records: Sequence[str | ItemMetadataRecord]) -> torch.Tensor:
        raise NotImplementedError


class BaseLLMSemanticProvider(ABC):
    @abstractmethod
    def score_cross_domain_pairs(
        self,
        source_items: Sequence[ItemMetadataRecord],
        target_items: Sequence[ItemMetadataRecord],
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def summarize_item_metadata(self, items: Sequence[ItemMetadataRecord]) -> list[str]:
        raise NotImplementedError
