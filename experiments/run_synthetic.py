from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cdr_framework.data import GraphBatch, InteractionBatch, ItemFeatures
from cdr_framework.framework import DualStructuralInjectionRecommender


def main() -> None:
    torch.manual_seed(42)
    model = DualStructuralInjectionRecommender.random_init(
        num_items=8,
        txt_dim=6,
        img_dim=6,
        attr_dim=4,
        hidden_dim=16,
        codebook_size=16,
        token_length=4,
        seed=42,
    )
    features = ItemFeatures(
        item_ids=torch.arange(8),
        id_vectors=torch.randn(8, 16),
        text_vectors=torch.randn(8, 6),
        image_vectors=torch.randn(8, 6),
        attr_vectors=torch.randn(8, 4),
        domains=("S", "S", "S", "S", "T", "T", "T", "T"),
    )
    graph = GraphBatch(
        source_edges=torch.tensor([[0, 1], [1, 2], [2, 3]]),
        target_edges=torch.tensor([[4, 5], [5, 6], [6, 7]]),
        cross_edges=torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]]),
        num_items=8,
    )
    batch = InteractionBatch(
        user_ids=torch.tensor([1, 2]),
        source_sequences=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        target_sequences=torch.tensor([[4, 5, 6], [5, 6, 7]]),
        positive_target_items=torch.tensor([7, 4]),
    )
    output = model.forward(batch, features, graph)
    print({name: round(value.item(), 4) for name, value in output.losses.items()})
    print(model.recommend(batch, features, graph, top_k=3, beam_size=4))


if __name__ == "__main__":
    main()
