from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from cdr_framework.losses import anti_collapse_variance_loss, context_alignment_loss, cpf_total_loss


@dataclass(frozen=True)
class CPFOutput:
    prediction: torch.Tensor
    losses: dict[str, torch.Tensor]


class ContextPredictionFeedback(nn.Module):
    def __init__(self, hidden_dim: int, target_dim: int, lambda_ctx: float = 0.1, lambda_var: float = 0.1):
        super().__init__()
        if hidden_dim <= 0 or target_dim <= 0:
            raise ValueError("hidden_dim and target_dim must be positive.")
        self.lambda_ctx = lambda_ctx
        self.lambda_var = lambda_var
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        target: torch.Tensor,
        reasoning_states: torch.Tensor,
    ) -> CPFOutput:
        prediction = self.predictor(torch.cat([latent, context], dim=-1))
        pred_loss = F.mse_loss(prediction, target)
        ctx_loss = context_alignment_loss(latent, context)
        var_loss = anti_collapse_variance_loss(reasoning_states)
        losses = cpf_total_loss(pred_loss, ctx_loss, var_loss, self.lambda_ctx, self.lambda_var)
        return CPFOutput(prediction=prediction, losses=losses)
