from __future__ import annotations

import torch
import torch.nn.functional as F


def orthogonality_loss(shared: torch.Tensor, specific: torch.Tensor) -> torch.Tensor:
    shared_n = F.normalize(shared, dim=-1)
    specific_n = F.normalize(specific, dim=-1)
    return torch.mean((shared_n * specific_n).sum(dim=-1).pow(2))


def shared_alignment_loss(source_shared: torch.Tensor, target_shared: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    count = min(source_shared.shape[0], target_shared.shape[0])
    if count == 0:
        return source_shared.new_tensor(0.0)
    logits = F.normalize(source_shared[:count], dim=-1) @ F.normalize(target_shared[:count], dim=-1).t() / temperature
    labels = torch.arange(count, device=source_shared.device)
    return F.cross_entropy(logits, labels)


def anti_collapse_variance_loss(states: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    flat = states.reshape(-1, states.shape[-1])
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-6)
    return torch.relu(gamma - std).mean()


def source_private_separation_loss(target_private: torch.Tensor, source_private_signal: torch.Tensor) -> torch.Tensor:
    return torch.mean((F.normalize(target_private, dim=-1) * F.normalize(source_private_signal, dim=-1)).sum(dim=-1).pow(2))


def context_alignment_loss(latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(F.normalize(latent, dim=-1), F.normalize(context, dim=-1), dim=-1).mean()


def cpf_total_loss(
    pred_loss: torch.Tensor,
    ctx_loss: torch.Tensor,
    var_loss: torch.Tensor,
    lambda_ctx: float = 0.1,
    lambda_var: float = 0.1,
) -> dict[str, torch.Tensor]:
    total = pred_loss + lambda_ctx * ctx_loss + lambda_var * var_loss
    return {"pred": pred_loss, "ctx": ctx_loss, "var": var_loss, "total": total}
