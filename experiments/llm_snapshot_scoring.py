"""Evaluation-only CPU catalog snapshots and explicitly ordered candidate scoring.

Not imported by the trainer: saved training code identities remain unchanged.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cdr_framework.ops import calibrate_scores
from cdr_framework.text_data import collate_text_rows


def map_tensors(value, operation):
    if isinstance(value, torch.Tensor):
        return operation(value)
    if isinstance(value, dict):
        return {key: map_tensors(entry, operation) for key, entry in value.items()}
    if value is None:
        return None
    raise TypeError('Snapshot may contain tensors, dictionaries and None only')


def tree_equal(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    return a is None and b is None


@torch.no_grad()
def cpu_item_snapshot(model):
    """Compute checkpoint-specific catalog features twice on one CPU thread.

    Move only item-side buffers/modules, never the Qwen backbone. Restore their
    original devices even on failure. No user histories or held-out labels enter
    this computation. Refuse a nonrepeatable snapshot rather than picking a pass.
    """
    model.eval()
    if model._evaluation_items is not None:
        raise RuntimeError('An evaluation catalog scope is already active')
    names = ('content_proj', 'code_proj', 'item_norm', 'graph_encoder', 'prototype')
    modules = {name: getattr(model, name) for name in names}
    devices = {name: next(module.parameters()).device for name, module in modules.items()}
    buffers = {name: model._buffers[name] for name in
               ('content', 'tokens', 'centroids', 'graph_shared', 'graph_source', 'graph_target')}
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for module in modules.values():
            module.cpu()
        for name, value in buffers.items():
            model._buffers[name] = value.cpu()
        first = map_tensors(model.item_states(), lambda t: t.detach().cpu().clone())
        second = model.item_states()
        if not tree_equal(first, second):
            raise RuntimeError('CPU catalog snapshot is not repeatable')
        def check(tensor):
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise RuntimeError('Catalog snapshot must be finite float32')
            return tensor
        map_tensors(first, check)
        return first
    finally:
        for name, module in modules.items():
            module.to(devices[name])
        for name, value in buffers.items():
            model._buffers[name] = value
        torch.set_num_threads(threads)


def _stable_order(scores, ids):
    if ids.ndim == 1:
        ids = ids.expand_as(scores)
    if scores.shape != ids.shape or scores.ndim != 2:
        raise ValueError('Expected matching [batch, candidates] scores and item IDs')
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise ValueError('Invalid score; only negative infinity masks are permitted')
    by_id = ids.argsort(dim=1, stable=True)
    by_score = scores.gather(1, by_id).argsort(dim=1, descending=True, stable=True)
    return by_id.gather(1, by_score)


def stable_topk(scores, ids, k):
    """Descending full-precision score, then ascending item ID; no label-based ties."""
    if ids.ndim == 1:
        ids = ids.expand_as(scores)
    order = _stable_order(scores, ids)[:, :k]
    ranked = ids.gather(1, order).masked_fill(~torch.isfinite(scores.gather(1, order)), 0)
    return F.pad(ranked, (0, max(0, k - ranked.shape[1])))


def rank_snapshot_parameters(cache, setting, k):
    pool = min(setting['pool'], cache['candidates'].shape[1])
    scores = cache['retrieval'][:, :pool]
    eligible = torch.isfinite(scores)
    if setting['weight'] > 0:
        generation = cache['generation'][:, :pool]
        if not torch.isfinite(generation[eligible]).all():
            raise ValueError('Nonfinite generation score')
        scores = calibrate_scores(scores, eligible, setting['normalization']) + setting['weight'] * calibrate_scores(
            generation, eligible, setting['normalization'])
    scores = scores.masked_fill(~eligible, -torch.inf)
    return stable_topk(scores, cache['candidates'][:, :pool], k)


@torch.no_grad()
def collect_snapshot_scores(model, rows, config, device, snapshot, existing=None, progress=None):
    """Same retrieval and SID generation scores, using a single signed catalog."""
    model.eval()
    if model._evaluation_items is not None:
        raise RuntimeError('An evaluation catalog scope is already active')
    items = map_tensors(snapshot, lambda t: t.to(device))
    model._evaluation_items = items
    try:
        vectors = F.normalize(items['base'], dim=-1)
        targets = model.target_ids
        k = min(len(targets), 1000)
        cache = existing or {key: [] for key in ('candidates', 'retrieval', 'generation', 'positive')}
        start = len(cache['positive'])
        if start > len(rows):
            raise RuntimeError('Cache has too many rows')
        arrays = {key: [torch.as_tensor(cache[key])] if start else [] for key in cache}
        loader = DataLoader(rows[start:], batch_size=config.evaluation_batch_size, collate_fn=collate_text_rows)
        completed = start
        for batch, seen in loader:
            batch = batch.to(device)
            state = model.encode(batch, items)
            scores = state['query'] @ vectors[targets].T / config.retrieval_temperature
            for row, history in enumerate(seen):
                indices = model.target_index[torch.tensor(sorted(history), device=device, dtype=torch.long)]
                scores[row, indices[indices >= 0]] = -torch.inf
            order = _stable_order(scores, targets)[:, :k]
            values = scores.gather(1, order)
            candidates = targets[order]
            flat = candidates.flatten()
            batch_indices = torch.arange(len(candidates), device=device).repeat_interleave(k)
            generation = torch.empty(len(flat), device=device)
            for offset in range(0, len(flat), config.decode_chunk_size):
                end = offset + config.decode_chunk_size
                generation[offset:end] = model.sequence_scores(state['prefix'][batch_indices[offset:end]], flat[offset:end])
            for key, value in (('candidates', candidates), ('retrieval', values),
                               ('generation', generation.reshape_as(values)), ('positive', batch.positive_items)):
                arrays[key].append(value.cpu())
            completed += len(candidates)
            if progress and (completed % 100 < len(candidates) or completed == len(rows)):
                progress({key: torch.cat(parts) for key, parts in arrays.items()}, completed)
        return {key: torch.cat(parts) for key, parts in arrays.items()}
    finally:
        model._evaluation_items = None
