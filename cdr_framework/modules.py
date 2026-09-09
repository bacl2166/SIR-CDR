from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from cdr_framework.data import GraphBatch
from cdr_framework.ops import build_symmetric_adjacency, sequence_mean


class MultimodalSemanticEncoder(nn.Module):
    def __init__(self, id_dim: int, txt_dim: int, img_dim: int, attr_dim: int, hidden_dim: int):
        super().__init__()
        self.id_proj = nn.Linear(id_dim, hidden_dim)
        self.text_proj = nn.Linear(txt_dim, hidden_dim)
        self.image_proj = nn.Linear(img_dim, hidden_dim)
        self.attr_proj = nn.Linear(attr_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())

    def forward(
        self,
        id_vectors: torch.Tensor,
        text_vectors: torch.Tensor,
        image_vectors: torch.Tensor,
        attr_vectors: torch.Tensor,
    ) -> torch.Tensor:
        id_sem = self.id_proj(id_vectors.float())
        modalities = torch.stack(
            [
                self.text_proj(text_vectors.float()),
                self.image_proj(image_vectors.float()),
                self.attr_proj(attr_vectors.float()),
            ],
            dim=1,
        )
        weights = torch.softmax(self.gate(torch.tanh(modalities)).squeeze(-1), dim=1).unsqueeze(-1)
        multimodal = (weights * modalities).sum(dim=1)
        return self.fuse(torch.cat([id_sem, multimodal], dim=-1))


class SharedSpecificGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, propagation_layers: int = 2, edge_feature_dim: int = 0):
        super().__init__()
        self.propagation_layers = propagation_layers
        self.edge_feature_dim = edge_feature_dim
        self.edge_confidence = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cd_proj = nn.LayerNorm(hidden_dim)
        self.sp_proj = nn.LayerNorm(hidden_dim)

    def forward(self, item_tokens: torch.Tensor, graph: GraphBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cd_edges = graph.cross_edges.to(item_tokens.device)
        source_edges = graph.source_edges.to(item_tokens.device)
        target_edges = graph.target_edges.to(item_tokens.device)
        cd_repr = self._propagate(item_tokens, cd_edges, graph.num_items, graph.cross_edge_features)
        source_repr = self._propagate(item_tokens, source_edges, graph.num_items, graph.source_edge_features)
        target_repr = self._propagate(item_tokens, target_edges, graph.num_items, graph.target_edge_features)
        return self.cd_proj(cd_repr), self.sp_proj(source_repr), self.sp_proj(target_repr)

    def _propagate(
        self,
        item_tokens: torch.Tensor,
        edges: torch.Tensor,
        num_items: int,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        adjacency = build_symmetric_adjacency(edges, num_items, item_tokens.device, item_tokens.dtype)
        if edges.numel() > 0:
            confidence_input = torch.cat([item_tokens[edges[:, 0]], item_tokens[edges[:, 1]]], dim=-1)
            if self.edge_feature_dim > 0:
                if edge_features is None:
                    edge_features = torch.zeros(
                        (edges.shape[0], self.edge_feature_dim),
                        device=item_tokens.device,
                        dtype=item_tokens.dtype,
                    )
                edge_features = edge_features.to(device=item_tokens.device, dtype=item_tokens.dtype)
                if edge_features.shape != (edges.shape[0], self.edge_feature_dim):
                    raise ValueError("edge_features must match [num_edges, edge_feature_dim].")
                confidence_input = torch.cat([confidence_input, edge_features], dim=-1)
            confidence = torch.sigmoid(self.edge_confidence(confidence_input)).reshape(-1)
            weighted = adjacency.clone()
            weighted[edges[:, 0], edges[:, 1]] *= confidence
            weighted[edges[:, 1], edges[:, 0]] *= confidence
            adjacency = weighted
        states = [item_tokens]
        current = item_tokens
        for _ in range(self.propagation_layers):
            current = adjacency @ current
            states.append(current)
        return torch.stack(states, dim=0).mean(dim=0)


class PrototypeDisentangler(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.shared = nn.Linear(hidden_dim, hidden_dim)
        self.source_specific = nn.Linear(hidden_dim, hidden_dim)
        self.target_specific = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, semantic: torch.Tensor, graph_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.base(torch.cat([semantic, graph_repr], dim=-1))
        return torch.tanh(self.shared(base)), torch.tanh(self.source_specific(base)), torch.tanh(self.target_specific(base))


class UserDisentangler(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.source_specific = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.target_specific = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())

    def forward(self, user_graph_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.shared(user_graph_context), self.source_specific(user_graph_context), self.target_specific(user_graph_context)


class CrossDomainStructuralInjector(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        shared_tokens: torch.Tensor,
        cd_graph_repr: torch.Tensor,
        source_history: torch.Tensor,
        target_history: torch.Tensor,
        user_shared: torch.Tensor | None = None,
    ) -> torch.Tensor:
        history = torch.cat([source_history, target_history], dim=1).long()
        values = self.value(torch.cat([shared_tokens, cd_graph_repr], dim=-1))
        gathered = values[history]
        if user_shared is None:
            user_shared = gathered.mean(dim=1)
        scores = (self.key(gathered) * self.query(user_shared).unsqueeze(1)).sum(dim=-1) / values.shape[-1] ** 0.5
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * gathered).sum(dim=1)


class SpecificDomainStructuralInjector(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.source_value = nn.Linear(hidden_dim * 2, hidden_dim)
        self.target_value = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_graph_repr: torch.Tensor,
        target_tokens: torch.Tensor,
        target_graph_repr: torch.Tensor,
        source_history: torch.Tensor,
        target_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_values = self.source_value(torch.cat([source_tokens, source_graph_repr], dim=-1))
        target_values = self.target_value(torch.cat([target_tokens, target_graph_repr], dim=-1))
        return sequence_mean(source_values, source_history.long()), sequence_mean(target_values, target_history.long())


class UserImplicitReasoner(nn.Module):
    def __init__(self, hidden_dim: int, reasoning_steps: int = 3):
        super().__init__()
        self.reasoning_steps = reasoning_steps
        gate_dim = hidden_dim * 4
        self.init = nn.Linear(hidden_dim * 3, hidden_dim)
        self.write_gate = nn.Linear(gate_dim, hidden_dim)
        self.read_gate = nn.Linear(gate_dim, hidden_dim)
        self.forget_gate = nn.Linear(gate_dim, hidden_dim)
        self.candidate = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.shared_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.private_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())

    @classmethod
    def random_init(cls, hidden_dim: int, reasoning_steps: int, seed: int) -> "UserImplicitReasoner":
        torch.manual_seed(seed)
        return cls(hidden_dim=hidden_dim, reasoning_steps=reasoning_steps)

    def forward(
        self,
        context: torch.Tensor,
        cd_signal: torch.Tensor,
        target_sp_signal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = torch.tanh(self.init(torch.cat([context, cd_signal, target_sp_signal], dim=-1)))
        states = []
        for _ in range(self.reasoning_steps):
            gate_input = torch.cat([state, context, cd_signal, target_sp_signal], dim=-1)
            write = torch.sigmoid(self.write_gate(gate_input))
            read = torch.sigmoid(self.read_gate(gate_input))
            forget = torch.sigmoid(self.forget_gate(gate_input))
            candidate = torch.tanh(self.candidate(torch.cat([read * state, context, cd_signal, target_sp_signal], dim=-1)))
            state = self.norm(forget * state + write * candidate)
            states.append(state)
        final = states[-1]
        shared = self.shared_head(torch.cat([final, cd_signal], dim=-1))
        private = self.private_head(torch.cat([final, target_sp_signal], dim=-1))
        return shared, private, torch.stack(states, dim=1)


class DualStructuralFusionPrefix(nn.Module):
    def __init__(self, hidden_dim: int, prefix_length: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prefix_length = prefix_length
        self.project = nn.Linear(hidden_dim * 5, prefix_length * hidden_dim)

    def forward(
        self,
        shared_latent: torch.Tensor,
        private_latent: torch.Tensor,
        cd_signal: torch.Tensor,
        target_sp_signal: torch.Tensor,
        codebook_summary: torch.Tensor,
    ) -> torch.Tensor:
        prefix = self.project(torch.cat([shared_latent, private_latent, cd_signal, target_sp_signal, codebook_summary], dim=-1))
        return prefix.reshape(prefix.shape[0], self.prefix_length, self.hidden_dim)


class AutoregressiveSemanticDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, prefix_length: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_id = vocab_size
        self.eos_id = vocab_size + 1
        self.embedding = nn.Embedding(vocab_size + 2, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size + 2)
        self.prefix_to_state = nn.Linear(prefix_length * hidden_dim, hidden_dim)

    def forward(self, prefix: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        initial = torch.tanh(self.prefix_to_state(prefix.flatten(start_dim=1))).unsqueeze(0)
        embeddings = self.embedding(input_tokens.long())
        outputs, _ = self.gru(embeddings, initial)
        return self.out(outputs)

    def next_log_probs(self, prefix: torch.Tensor, generated_tokens: list[int]) -> torch.Tensor:
        device = prefix.device
        tokens = torch.tensor([generated_tokens], dtype=torch.long, device=device)
        logits = self(prefix[:1], tokens)[:, -1, :]
        return F.log_softmax(logits, dim=-1).squeeze(0)
