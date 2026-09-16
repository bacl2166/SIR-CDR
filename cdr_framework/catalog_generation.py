"""Label-free, catalog-constrained semantic generation for text-only SIR-CDR."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


class CatalogTrie:
    """Index fixed-length token sequences from global ``semantic_ids[item_id]``.

    ``item_ids`` is a one-dimensional catalog subset; ``semantic_ids`` is the
    full [N, L] integer table, including padding row 0. Item 0 is excluded,
    while semantic token 0 remains legal. Repeated item IDs are deduplicated.
    Codes are used verbatim, without per-depth offsets or special tokens.
    """

    def __init__(self, item_ids: Tensor, semantic_ids: Tensor):
        if not isinstance(item_ids, Tensor) or item_ids.ndim != 1:
            raise ValueError("item_ids must have shape [num_catalog_items].")
        if not isinstance(semantic_ids, Tensor) or semantic_ids.ndim != 2:
            raise ValueError("semantic_ids must be the full global table [N, L].")
        if item_ids.dtype not in _INTEGER_DTYPES or semantic_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("item_ids and semantic_ids must have integer dtypes.")
        if semantic_ids.shape[1] < 1:
            raise ValueError("semantic_ids must have positive token length.")
        ids = sorted(set(item_ids.detach().cpu().tolist()))
        if any(item < 0 or item >= semantic_ids.shape[0] for item in ids):
            raise ValueError("item_ids must index rows of the global semantic_ids table.")
        self.item_ids = tuple(item for item in ids if item != 0)
        self.token_length = int(semantic_ids.shape[1])
        self.num_global_items = int(semantic_ids.shape[0])
        children: dict[tuple[int, ...], set[int]] = {}
        terminals: dict[tuple[int, ...], list[int]] = {}
        rows = semantic_ids[list(self.item_ids)].detach().cpu().tolist()
        for item, row in zip(self.item_ids, rows):
            sequence = tuple(row)
            if any(code < 0 for code in sequence):
                raise ValueError("Catalog items must have nonnegative semantic token IDs.")
            for depth, code in enumerate(sequence):
                children.setdefault(sequence[:depth], set()).add(code)
            terminals.setdefault(sequence, []).append(item)
        self._children = {key: tuple(sorted(value)) for key, value in children.items()}
        self._terminals = {key: tuple(value) for key, value in terminals.items()}
        self._codes = frozenset(code for sequence in terminals for code in sequence)

    def next_codes(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        """Return sorted legal next codes, or () for terminal/unknown prefixes."""
        return self._children.get(tuple(prefix), ())

    def terminal_items(self, sequence: tuple[int, ...]) -> tuple[int, ...]:
        """Return all distinct catalog item IDs sharing a complete sequence."""
        return self._terminals.get(tuple(sequence), ())


@torch.no_grad()
def constrained_generate(
    decoder,
    prefix: Tensor,
    trie: CatalogTrie,
    beam_size: int,
    top_k: int,
    seen_items: list[set[int]],
    query: Tensor,
    item_vectors: Tensor,
) -> Tensor:
    """Return [B, top_k] global item IDs (long, prefix device), padded with 0.

    ``decoder.forward(prefix, input_tokens)`` returns [beam, T, V] logits and
    exposes integer ``bos_id``/``eos_id``. ``prefix`` is [B, P, H], ``query``
    is [B, D], and ``item_vectors`` is the global [N, D] table. All three
    tensors must be floating point on the same device. Callers control the
    decoder's train/eval mode; this function disables gradients only.

    Scores sum full-vocabulary log probabilities of L codes plus EOS, with
    no length penalty or renormalization over legal branches. Each user gets
    at most beam_size prefixes per intermediate depth; all last-code branches
    receive EOS scores before the final beam is selected. Active prefixes are
    evaluated in a batch per user/depth.

    Sequences sort by descending score, then lexicographic code sequence on
    exact ties. Within each sequence only, unseen items sort by descending
    query-item cosine, then ascending item ID. Seen filtering does not refill
    the beam; there is no retrieval fallback. Neither labels nor targets are
    accepted or consulted.
    """
    for name, value in (("beam_size", beam_size), ("top_k", top_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(trie, CatalogTrie):
        raise ValueError("trie must be a CatalogTrie.")
    for name, value, ndim in (("prefix", prefix, 3), ("query", query, 2),
                               ("item_vectors", item_vectors, 2)):
        if not isinstance(value, Tensor) or value.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions.")
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point.")
    batch_size = prefix.shape[0]
    if prefix.shape[1] == 0 or prefix.shape[2] == 0:
        raise ValueError("prefix must have positive prefix length and hidden dimension.")
    if query.shape[0] != batch_size or query.shape[1] == 0:
        raise ValueError("query must have shape [B, D] with positive D.")
    if item_vectors.shape != (trie.num_global_items, query.shape[1]):
        raise ValueError("item_vectors must be the full global table [N, D].")
    if query.device != prefix.device or item_vectors.device != prefix.device:
        raise ValueError("prefix, query, and item_vectors must share a device.")
    if not isinstance(seen_items, list) or len(seen_items) != batch_size or any(
        not isinstance(items, set) for items in seen_items
    ):
        raise ValueError("seen_items must contain one set per user.")
    bos_id, eos_id = decoder.bos_id, decoder.eos_id
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0
           for token in (bos_id, eos_id)) or bos_id == eos_id:
        raise ValueError("decoder bos_id/eos_id must be distinct nonnegative integers.")
    if bos_id in trie._codes or eos_id in trie._codes:
        raise ValueError("Catalog codes must not contain BOS or EOS.")
    result = torch.zeros((batch_size, top_k), dtype=torch.long, device=prefix.device)
    if not trie.item_ids or batch_size == 0:
        return result

    def next_log_probs(user: int, sequences: list[tuple[int, ...]]) -> Tensor:
        tokens = torch.tensor([(bos_id, *sequence) for sequence in sequences],
                              dtype=torch.long, device=prefix.device)
        logits = decoder(prefix[user:user + 1].expand(len(sequences), -1, -1), tokens)
        if not isinstance(logits, Tensor) or logits.ndim != 3 or logits.shape[:2] != tokens.shape:
            raise ValueError("decoder must return logits of shape [beam, T, V].")
        if not logits.is_floating_point():
            raise ValueError("decoder logits must be floating point.")
        if max(bos_id, eos_id, max(trie._codes)) >= logits.shape[-1]:
            raise ValueError("Catalog/special token IDs exceed the decoder vocabulary.")
        # Normalize before selecting legal tokens, retaining illegal-token mass.
        return F.log_softmax(logits[:, -1, :], dim=-1)

    for user in range(batch_size):
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        for depth in range(trie.token_length):
            probabilities = next_log_probs(user, [sequence for sequence, _ in beams])
            candidates = []
            for row, (sequence, score) in enumerate(beams):
                codes = trie.next_codes(sequence)
                values = probabilities[row, list(codes)].tolist()
                candidates.extend(((*sequence, code), score + value)
                                  for code, value in zip(codes, values))
            if depth == trie.token_length - 1:
                eos_probs = next_log_probs(user, [sequence for sequence, _ in candidates])
                candidates = [(sequence, score + eos)
                              for (sequence, score), eos in
                              zip(candidates, eos_probs[:, eos_id].tolist())]
            candidates.sort(key=lambda entry: (-entry[1], entry[0]))
            beams = candidates[:beam_size]

        ranked: list[int] = []
        emitted: set[int] = set()
        for sequence, _ in beams:
            items = [item for item in trie.terminal_items(sequence)
                     if item not in seen_items[user] and item not in emitted]
            if not items:
                continue
            cosine = F.cosine_similarity(query[user:user + 1], item_vectors[items], dim=-1)
            ordered = sorted(zip(items, cosine.tolist()), key=lambda entry: (-entry[1], entry[0]))
            for item, _ in ordered:
                ranked.append(item)
                emitted.add(item)
                if len(ranked) == top_k:
                    break
            if len(ranked) == top_k:
                break
        if ranked:
            result[user, :len(ranked)] = torch.tensor(ranked, dtype=torch.long, device=prefix.device)
    return result
