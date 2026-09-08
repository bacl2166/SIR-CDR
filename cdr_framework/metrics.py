from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T", int, str)


def hit_rate_at_k(ranked_items: Sequence[Sequence[T]], positives: Sequence[T], k: int) -> float:
    _validate_metric_inputs(ranked_items, positives, k)
    hits = [positive in list(items)[:k] for items, positive in zip(ranked_items, positives)]
    return sum(hits) / len(hits) if hits else 0.0


def ndcg_at_k(ranked_items: Sequence[Sequence[T]], positives: Sequence[T], k: int) -> float:
    _validate_metric_inputs(ranked_items, positives, k)
    scores = []
    for items, positive in zip(ranked_items, positives):
        top_items = list(items)[:k]
        if positive not in top_items:
            scores.append(0.0)
            continue
        rank = top_items.index(positive) + 1
        scores.append(1.0 / math.log2(rank + 1))
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(ranked_items: Sequence[Sequence[T]], positives: Sequence[T], k: int) -> float:
    _validate_metric_inputs(ranked_items, positives, k)
    reciprocal_ranks = []
    for items, positive in zip(ranked_items, positives):
        top_items = list(items)[:k]
        if positive not in top_items:
            reciprocal_ranks.append(0.0)
            continue
        reciprocal_ranks.append(1.0 / (top_items.index(positive) + 1))
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def coverage(ranked_items: Sequence[Sequence[T]], total_items: int) -> float:
    if total_items <= 0:
        raise ValueError("total_items must be positive.")
    recommended = {item for row in ranked_items for item in row}
    return len(recommended) / total_items


def token_lookup_hit_rate(lookup_hits: Sequence[bool]) -> float:
    if not lookup_hits:
        return 0.0
    return sum(1 for hit in lookup_hits if hit) / len(lookup_hits)


def _validate_metric_inputs(ranked_items: Sequence[Sequence[T]], positives: Sequence[T], k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive.")
    if len(ranked_items) != len(positives):
        raise ValueError("ranked_items and positives must have the same length.")
