from __future__ import annotations

from collections import Counter

import torch


def build_transition_edges(sequences: torch.Tensor, num_items: int) -> tuple[torch.Tensor, torch.Tensor]:
    if sequences.ndim != 2:
        raise ValueError("sequences must have shape [batch, length].")
    if num_items <= 0:
        raise ValueError("num_items must be positive.")

    counts: Counter[tuple[int, int]] = Counter()
    for row in sequences.long().tolist():
        for src, dst in zip(row[:-1], row[1:]):
            if src == dst:
                continue
            if src < 0 or dst < 0 or src >= num_items or dst >= num_items:
                raise ValueError("sequence item id is outside num_items.")
            edge = (min(src, dst), max(src, dst))
            counts[edge] += 1

    ordered_edges = sorted(counts)
    if not ordered_edges:
        return torch.empty((0, 2), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    edge_tensor = torch.tensor(ordered_edges, dtype=torch.long)
    count_tensor = torch.tensor([float(counts[edge]) for edge in ordered_edges], dtype=torch.float32)
    return edge_tensor, count_tensor
