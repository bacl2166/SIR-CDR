from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from cdr_framework.losses import orthogonality_loss, source_private_separation_loss
from cdr_framework.modules import (
    AutoregressiveSemanticDecoder,
    DualStructuralFusionPrefix,
    UserDisentangler,
    UserImplicitReasoner,
)
from cdr_framework.modules_cpf import ContextPredictionFeedback


@dataclass(frozen=True)
class RecommendationRow:
    source_items: tuple[int, ...]
    target_items: tuple[int, ...]
    positive_target_item: int


@dataclass(frozen=True)
class FormalBatch:
    source_items: torch.Tensor
    source_lengths: torch.Tensor
    target_items: torch.Tensor
    target_lengths: torch.Tensor
    positive_items: torch.Tensor

    def to(self, device: torch.device | str) -> "FormalBatch":
        return FormalBatch(*(value.to(device) for value in self.__dict__.values()))


@dataclass(frozen=True)
class FormalOutput:
    prefix: torch.Tensor
    query: torch.Tensor
    logits: torch.Tensor
    losses: dict[str, torch.Tensor]


@dataclass(frozen=True)
class FixedCatalog:
    item_latents: torch.Tensor
    semantic_ids: torch.Tensor
    codebooks: torch.Tensor
    target_item_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.item_latents.ndim != 2:
            raise ValueError("item_latents must have shape [num_items + 1, dimension].")
        if self.semantic_ids.ndim != 2:
            raise ValueError("semantic_ids must have shape [num_items + 1, token_length].")
        if self.codebooks.ndim != 3:
            raise ValueError("codebooks must have shape [token_length, codebook_size, dimension].")
        if len(self.item_latents) != len(self.semantic_ids):
            raise ValueError("Item latent and Semantic ID row counts must match.")
        if self.semantic_ids.shape[1] != self.codebooks.shape[0]:
            raise ValueError("Semantic ID length must match the number of codebook levels.")


def load_recommendation_rows(
    path: str | Path,
    *,
    max_sequence_length: int,
) -> list[RecommendationRow]:
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive.")
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            source = tuple(int(item) for item in payload["source_items"])[-max_sequence_length:]
            target = tuple(int(item) for item in payload["target_items"])[-max_sequence_length:]
            if not source or not target:
                raise ValueError(f"Empty history at {path}:{line_number}.")
            rows.append(
                RecommendationRow(
                    source_items=source,
                    target_items=target,
                    positive_target_item=int(payload["positive_target_item"]),
                )
            )
    return rows


def collate_recommendation_rows(rows: Sequence[RecommendationRow]) -> FormalBatch:
    if not rows:
        raise ValueError("Cannot collate an empty recommendation batch.")

    def pad(histories: Sequence[tuple[int, ...]]) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(history) for history in histories], dtype=torch.long)
        values = torch.zeros((len(histories), int(lengths.max())), dtype=torch.long)
        for index, history in enumerate(histories):
            values[index, : len(history)] = torch.tensor(history, dtype=torch.long)
        return values, lengths

    source, source_lengths = pad([row.source_items for row in rows])
    target, target_lengths = pad([row.target_items for row in rows])
    return FormalBatch(
        source_items=source,
        source_lengths=source_lengths,
        target_items=target,
        target_lengths=target_lengths,
        positive_items=torch.tensor(
            [row.positive_target_item for row in rows], dtype=torch.long
        ),
    )


class SIRCDRRecommender(nn.Module):
    def __init__(
        self,
        catalog: FixedCatalog,
        *,
        hidden_dim: int,
        reasoning_steps: int = 3,
        prefix_length: int = 4,
        retrieval_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or retrieval_temperature <= 0:
            raise ValueError("hidden_dim and retrieval_temperature must be positive.")
        self.hidden_dim = hidden_dim
        self.token_length = int(catalog.semantic_ids.shape[1])
        self.codebook_size = int(catalog.codebooks.shape[1])
        self.retrieval_temperature = retrieval_temperature

        safe_tokens = catalog.semantic_ids.clamp_min(0).long()
        semantic_vectors = torch.zeros_like(catalog.item_latents.float())
        for level in range(self.token_length):
            semantic_vectors += catalog.codebooks[level][safe_tokens[:, level]].float()
        semantic_vectors[0].zero_()

        self.register_buffer("item_latents", catalog.item_latents.float(), persistent=False)
        self.register_buffer("semantic_ids", catalog.semantic_ids.long(), persistent=False)
        self.register_buffer("semantic_vectors", semantic_vectors, persistent=False)
        self.register_buffer("target_item_ids", catalog.target_item_ids.long(), persistent=False)
        candidate_index = torch.full((len(catalog.item_latents),), -1, dtype=torch.long)
        candidate_index[catalog.target_item_ids.long()] = torch.arange(len(catalog.target_item_ids))
        self.register_buffer("candidate_index", candidate_index, persistent=False)

        input_dim = int(catalog.item_latents.shape[1])
        self.content_projection = nn.Linear(input_dim, hidden_dim)
        self.semantic_projection = nn.Linear(input_dim, hidden_dim)
        self.item_norm = nn.LayerNorm(hidden_dim)
        self.source_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.target_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.user_disentangler = UserDisentangler(hidden_dim)
        self.reasoner = UserImplicitReasoner(hidden_dim, reasoning_steps)
        self.prefix = DualStructuralFusionPrefix(hidden_dim, prefix_length)
        self.decoder = AutoregressiveSemanticDecoder(
            self.codebook_size, hidden_dim, prefix_length
        )
        self.cpf = ContextPredictionFeedback(hidden_dim, hidden_dim)

    def item_representations(self) -> torch.Tensor:
        values = self.item_norm(
            self.content_projection(self.item_latents)
            + self.semantic_projection(self.semantic_vectors)
        )
        return values.index_fill(0, torch.tensor([0], device=values.device), 0.0)

    def _encode_sequence(
        self,
        item_repr: torch.Tensor,
        items: torch.Tensor,
        lengths: torch.Tensor,
        encoder: nn.GRU,
    ) -> torch.Tensor:
        embedded = item_repr[items.long()]
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = encoder(packed)
        return hidden[-1]

    def encode_user(
        self, batch: FormalBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item_repr = self.item_representations()
        source = self._encode_sequence(
            item_repr, batch.source_items, batch.source_lengths, self.source_encoder
        )
        target = self._encode_sequence(
            item_repr, batch.target_items, batch.target_lengths, self.target_encoder
        )
        context = 0.5 * (source + target)
        user_shared, source_specific, target_specific = self.user_disentangler(context)
        shared, private, reasoning_states = self.reasoner(
            context, user_shared + source, target_specific + target
        )
        history_summary = 0.5 * (source_specific + target_specific)
        prefix = self.prefix(shared, private, source, target, history_summary)
        return prefix, shared, private, context, reasoning_states

    def forward(self, batch: FormalBatch) -> FormalOutput:
        prefix, shared, private, context, reasoning_states = self.encode_user(batch)
        target_tokens = self.semantic_ids[batch.positive_items.long()]
        bos = torch.full(
            (len(target_tokens), 1),
            self.decoder.bos_id,
            dtype=torch.long,
            device=target_tokens.device,
        )
        eos = torch.full_like(bos, self.decoder.eos_id)
        decoder_input = torch.cat([bos, target_tokens], dim=1)
        decoder_target = torch.cat([target_tokens, eos], dim=1)
        logits = self.decoder(prefix, decoder_input)
        generation = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), decoder_target.reshape(-1)
        )

        item_repr = self.item_representations()
        positive_repr = item_repr[batch.positive_items.long()]
        cpf = self.cpf(shared + private, context, positive_repr, reasoning_states)
        query = F.normalize(cpf.prediction, dim=-1)
        candidates = F.normalize(item_repr[self.target_item_ids], dim=-1)
        retrieval_logits = query @ candidates.t() / self.retrieval_temperature
        labels = self.candidate_index[batch.positive_items.long()]
        if torch.any(labels < 0):
            raise ValueError("A positive item is outside the target catalog.")
        retrieval = F.cross_entropy(retrieval_logits, labels)
        orthogonality = orthogonality_loss(shared, private)
        separation = source_private_separation_loss(private, context)
        losses = {
            "generation": generation,
            "retrieval": retrieval,
            "cpf": cpf.losses["total"],
            "orthogonality": orthogonality,
            "separation": separation,
        }
        losses["total"] = (
            generation
            + retrieval
            + 0.1 * cpf.losses["total"]
            + 0.05 * orthogonality
            + 0.05 * separation
        )
        return FormalOutput(prefix=prefix, query=query, logits=logits, losses=losses)

    @torch.no_grad()
    def rank_full_catalog(
        self,
        batch: FormalBatch,
        *,
        top_k: int,
        candidate_chunk_size: int = 512,
        rerank_candidates: int = 200,
        generation_weight: float = 1.0,
        retrieval_weight: float = 1.0,
    ) -> torch.Tensor:
        if top_k <= 0 or candidate_chunk_size <= 0 or rerank_candidates <= 0:
            raise ValueError("Ranking sizes must be positive.")
        prefix, shared, private, context, states = self.encode_user(batch)
        item_repr = self.item_representations()
        del states
        query = F.normalize(
            self.cpf.predictor(torch.cat([shared + private, context], dim=-1)), dim=-1
        )
        rerank_count = min(
            max(top_k, rerank_candidates), len(self.target_item_ids)
        )
        best_scores = torch.empty((len(query), 0), device=query.device)
        best_items = torch.empty((len(query), 0), dtype=torch.long, device=query.device)
        normalized_items = F.normalize(item_repr, dim=-1)
        for start in range(0, len(self.target_item_ids), candidate_chunk_size):
            chunk_items = self.target_item_ids[start : start + candidate_chunk_size]
            scores = (
                query @ normalized_items[chunk_items].t()
            ) / self.retrieval_temperature
            for row in range(len(query)):
                seen = batch.target_items[row, : batch.target_lengths[row]].tolist()
                for seen_item in seen:
                    matches = (chunk_items == seen_item).nonzero(as_tuple=False)
                    if len(matches):
                        scores[row, matches[0, 0]] = -torch.inf
            expanded_items = chunk_items[None].expand(len(query), -1)
            merged_scores = torch.cat([best_scores, scores], dim=1)
            merged_items = torch.cat([best_items, expanded_items], dim=1)
            keep = min(rerank_count, merged_scores.shape[1])
            best_scores, keep_indices = torch.topk(merged_scores, keep, dim=1)
            best_items = merged_items.gather(1, keep_indices)

        retrieval_scores = best_scores
        item_ids = best_items
        tokens = self.semantic_ids[item_ids]
        batch_size = len(query)
        expanded_prefix = prefix[:, None].expand(-1, rerank_count, -1, -1).reshape(
            batch_size * rerank_count, prefix.shape[1], prefix.shape[2]
        )
        expanded_tokens = tokens.reshape(batch_size * rerank_count, self.token_length)
        bos = torch.full(
            (len(expanded_tokens), 1),
            self.decoder.bos_id,
            dtype=torch.long,
            device=query.device,
        )
        eos = torch.full_like(bos, self.decoder.eos_id)
        logits = self.decoder(expanded_prefix, torch.cat([bos, expanded_tokens], dim=1))
        targets = torch.cat([expanded_tokens, eos], dim=1)
        generation = F.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1).mean(dim=1).reshape(batch_size, rerank_count)
        scores = generation_weight * generation + retrieval_weight * retrieval_scores
        _, order = torch.topk(scores, min(top_k, rerank_count), dim=1)
        return item_ids.gather(1, order)
