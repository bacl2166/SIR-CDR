from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_mean(embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gathered = embeddings[indices]
    return gathered.mean(dim=1)


def cosine_tie_score(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    query = query.reshape(1, -1)
    return F.cosine_similarity(query, candidates, dim=-1)


def build_symmetric_adjacency(
    edges: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    adjacency = torch.eye(num_nodes, device=device, dtype=dtype)
    if edges.numel() == 0:
        return adjacency
    edges = edges.to(device=device, dtype=torch.long)
    src, dst = edges[:, 0], edges[:, 1]
    adjacency[src, dst] = 1.0
    adjacency[dst, src] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    norm = torch.rsqrt(degree).reshape(-1, 1) * adjacency * torch.rsqrt(degree).reshape(1, -1)
    return norm
