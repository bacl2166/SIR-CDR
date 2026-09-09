from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TokenizerOutput:
    universal: torch.Tensor
    domain_specific: torch.Tensor
    gate: torch.Tensor
    final: torch.Tensor


class DomainAdaptiveSemanticTokenizer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_domains: int):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_domains <= 0:
            raise ValueError("input_dim, hidden_dim, and num_domains must be positive.")
        self.domain_embedding = nn.Embedding(num_domains, hidden_dim)
        self.universal = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.domain_specific = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.gate = nn.Linear(hidden_dim * 2, 1)

    def forward(self, vectors: torch.Tensor, domains: torch.Tensor) -> TokenizerOutput:
        if vectors.ndim != 2:
            raise ValueError("vectors must have shape [num_items, input_dim].")
        domains = domains.long().to(vectors.device)
        domain_context = self.domain_embedding(domains)
        universal = self.universal(vectors.float())
        domain_specific = self.domain_specific(torch.cat([vectors.float(), domain_context], dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat([universal, domain_specific], dim=-1)))
        final = gate * universal + (1.0 - gate) * domain_specific
        return TokenizerOutput(universal=universal, domain_specific=domain_specific, gate=gate, final=final)
