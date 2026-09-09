from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ItemFeatures:
    item_ids: torch.Tensor
    id_vectors: torch.Tensor
    text_vectors: torch.Tensor
    image_vectors: torch.Tensor
    attr_vectors: torch.Tensor
    domains: Sequence[str]

    def target_item_ids(self) -> torch.Tensor:
        mask = torch.tensor([domain == "T" for domain in self.domains], dtype=torch.bool)
        return self.item_ids[mask]


@dataclass(frozen=True)
class GraphBatch:
    source_edges: torch.Tensor
    target_edges: torch.Tensor
    cross_edges: torch.Tensor
    num_items: int
    source_edge_features: torch.Tensor | None = None
    target_edge_features: torch.Tensor | None = None
    cross_edge_features: torch.Tensor | None = None


@dataclass(frozen=True)
class InteractionBatch:
    user_ids: torch.Tensor
    source_sequences: torch.Tensor
    target_sequences: torch.Tensor
    positive_target_items: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.user_ids.shape[0])


@dataclass(frozen=True)
class ForwardOutput:
    prefix: torch.Tensor
    target_tokens: torch.Tensor
    logits: torch.Tensor
    losses: dict[str, torch.Tensor]
