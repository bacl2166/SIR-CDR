from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError("lengths must be one-dimensional")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if (lengths < 1).any() or (lengths > max_length).any():
        raise ValueError("lengths must be within the padded sequence width")
    return torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def sequence_mean(
    embeddings: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    gathered = embeddings[indices]
    if lengths is None:
        return gathered.mean(dim=1)
    mask = sequence_mask(lengths.to(indices.device), indices.shape[1]).unsqueeze(-1)
    return (gathered * mask).sum(dim=1) / lengths.to(gathered).unsqueeze(1)


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
