from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch

from cdr_framework.data import GraphBatch, InteractionBatch, ItemFeatures


class BaseDataProvider(ABC):
    """Reserved data boundary for real datasets, graph builders, and samplers."""

    @abstractmethod
    def iter_train_batches(self) -> Iterable[InteractionBatch]:
        raise NotImplementedError

    @abstractmethod
    def item_features(self) -> ItemFeatures:
        raise NotImplementedError

    @abstractmethod
    def item_graph(self) -> GraphBatch:
        raise NotImplementedError


class BaseSemanticBackbone(ABC):
    """Reserved model boundary for LLM, text encoder, image encoder, or API calls."""

    @abstractmethod
    def encode_text(self, raw_text: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_image(self, image_refs: list[str]) -> torch.Tensor:
        raise NotImplementedError


class IdentitySemanticBackbone(BaseSemanticBackbone):
    """Placeholder adapter used when features are already precomputed."""

    def encode_text(self, raw_text: list[str]) -> torch.Tensor:
        raise RuntimeError("Text encoding is reserved for an external semantic backbone.")

    def encode_image(self, image_refs: list[str]) -> torch.Tensor:
        raise RuntimeError("Image encoding is reserved for an external semantic backbone.")
