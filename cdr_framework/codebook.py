from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from cdr_framework.ops import cosine_tie_score


@dataclass(frozen=True)
class BeamResult:
    tokens: list[int]
    score: float


@dataclass(frozen=True)
class RecommendationHit:
    item_id: int | str
    score: float
    tokens: tuple[int, ...]


class SemanticCodebook(torch.nn.Module):
    def __init__(self, embeddings: torch.Tensor):
        super().__init__()
        self.embeddings = torch.nn.Parameter(embeddings.clone().detach().float())

    @classmethod
    def random_init(cls, codebook_size: int, hidden_dim: int, seed: int) -> "SemanticCodebook":
        generator = torch.Generator().manual_seed(seed)
        embeddings = torch.randn(codebook_size, hidden_dim, generator=generator) * 0.02
        return cls(embeddings)

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    def quantize(self, vectors: torch.Tensor, token_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = vectors
        tokens = []
        reconstruction = torch.zeros_like(vectors)
        for _ in range(token_length):
            distances = torch.cdist(residual, self.embeddings)
            token = torch.argmin(distances, dim=1)
            chosen = self.embeddings[token]
            tokens.append(token)
            reconstruction = reconstruction + chosen
            residual = residual - chosen
        token_tensor = torch.stack(tokens, dim=1)
        loss = F.mse_loss(reconstruction, vectors.detach()) + 0.25 * F.mse_loss(reconstruction.detach(), vectors)
        return token_tensor, reconstruction, loss

    def pooled(self, query: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(query @ self.embeddings.t(), dim=-1)
        return weights @ self.embeddings


class ItemTokenIndex:
    def __init__(self) -> None:
        self._entries: dict[tuple[int, ...], list[tuple[int | str, torch.Tensor]]] = {}

    def add(self, item_id: int | str, tokens: list[int] | torch.Tensor, vector: torch.Tensor) -> None:
        key = tuple(int(v) for v in tokens)
        self._entries.setdefault(key, []).append((item_id, vector.detach().float()))

    def clear(self) -> None:
        self._entries.clear()

    def lookup_ranked(
        self,
        tokens: list[int] | torch.Tensor,
        sequence_score: float,
        query_vector: torch.Tensor,
        top_k: int,
        tie_penalty: float = 0.1,
    ) -> list[RecommendationHit]:
        key = tuple(int(v) for v in tokens)
        entries = self._entries.get(key, [])
        if not entries:
            return []
        vectors = torch.stack([entry[1] for entry in entries])
        tie_scores = cosine_tie_score(query_vector.detach().float(), vectors)
        hits = [
            RecommendationHit(
                item_id=item_id,
                score=float(sequence_score + tie_penalty * tie_scores[idx].item()),
                tokens=key,
            )
            for idx, (item_id, _) in enumerate(entries)
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    def nearest_ranked(self, query_vector: torch.Tensor, top_k: int) -> list[RecommendationHit]:
        flat_entries = [
            (tokens, item_id, vector)
            for tokens, entries in self._entries.items()
            for item_id, vector in entries
        ]
        if not flat_entries:
            return []
        vectors = torch.stack([entry[2] for entry in flat_entries])
        scores = cosine_tie_score(query_vector.detach().float(), vectors)
        hits = [
            RecommendationHit(
                item_id=item_id,
                score=float(scores[idx].item()),
                tokens=tokens,
            )
            for idx, (tokens, item_id, _) in enumerate(flat_entries)
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]


def beam_search(
    step_log_probs: Callable[[list[int]], dict[int, float]],
    bos_id: int,
    eos_id: int,
    beam_size: int,
    max_length: int,
    length_penalty: float = 0.7,
) -> list[BeamResult]:
    beams: list[tuple[list[int], float, bool]] = [([bos_id], 0.0, False)]
    completed: list[BeamResult] = []
    for _ in range(max_length):
        candidates: list[tuple[list[int], float, bool]] = []
        for prefix, score, done in beams:
            if done:
                candidates.append((prefix, score, True))
                continue
            for token, log_prob in step_log_probs(prefix).items():
                new_prefix = prefix + [int(token)]
                candidates.append((new_prefix, score + float(log_prob), token == eos_id))
        candidates.sort(key=lambda item: _normalized_score(item[1], item[0], bos_id, eos_id, length_penalty), reverse=True)
        beams = candidates[:beam_size]
        for prefix, score, done in beams:
            if done:
                completed.append(BeamResult(tokens=_strip_special(prefix, bos_id, eos_id), score=score))
        if completed:
            break
    if not completed:
        completed = [BeamResult(tokens=_strip_special(prefix, bos_id, eos_id), score=score) for prefix, score, _ in beams]
    completed.sort(key=lambda item: item.score / max(len(item.tokens), 1) ** length_penalty, reverse=True)
    return completed[:beam_size]


def _strip_special(tokens: list[int], bos_id: int, eos_id: int) -> list[int]:
    stripped = [token for token in tokens if token not in {bos_id, eos_id}]
    return stripped


def _normalized_score(score: float, tokens: list[int], bos_id: int, eos_id: int, length_penalty: float) -> float:
    length = max(len(_strip_special(tokens, bos_id, eos_id)), 1)
    return score / (length**length_penalty)
