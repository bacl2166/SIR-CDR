from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch


@dataclass(frozen=True)
class GraphEdgeFeatures:
    edge_type_id: int = 0
    semantic_score: float = 0.0
    visual_score: float = 0.0
    category_score: float = 0.0
    co_user_count: float = 0.0
    transition_count: float = 0.0
    llm_score: float = 0.0

    @classmethod
    def width(cls) -> int:
        return 7

    def as_tuple(self) -> tuple[float, ...]:
        return (
            float(self.edge_type_id),
            self.semantic_score,
            self.visual_score,
            self.category_score,
            self.co_user_count,
            self.transition_count,
            self.llm_score,
        )


def edge_feature_tensor(
    features: Sequence[GraphEdgeFeatures],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not features:
        return torch.empty((0, GraphEdgeFeatures.width()), device=device, dtype=dtype)
    return torch.tensor([feature.as_tuple() for feature in features], device=device, dtype=dtype)
