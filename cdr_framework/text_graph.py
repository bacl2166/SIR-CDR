"""Training-history graphs and text-feature-only sparse graph propagation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn


_PATHS = ("shared", "source", "target")
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _integer(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _normalized_graph(edges: set[tuple[int, int]], num_items: int) -> Tensor:
    edges = edges | {(i, i) for i in range(1, num_items)}
    indices = torch.tensor(sorted(edges), dtype=torch.long).reshape(-1, 2).t()
    degree = torch.bincount(indices[0], minlength=num_items).to(torch.float32)
    inverse = degree.clamp_min(1).rsqrt()
    values = inverse[indices[0]] * inverse[indices[1]]
    return torch.sparse_coo_tensor(
        indices, values, (num_items, num_items), check_invariants=True,
    ).coalesce()


def build_training_graphs(
    train_path: Path,
    num_items: int,
    target_item_ids: Tensor,
    cross_window: int = 5,
) -> dict[str, Tensor]:
    """Return CPU float32 coalesced COO graphs of shape ``(num_items, num_items)``.

    IDs are global catalog indices: 0 is padding; target IDs are explicitly
    listed, and all remaining nonzero IDs belong to the source domain. Every
    row is validated. Timestamps must be finite numbers; the greatest timestamp
    per user wins, with the last file row breaking ties. Only that row's
    histories contribute edges, never its positive label. A previous label may
    legitimately appear in a later training history.

    This is a static training graph: only train_path is read, with no validation
    or test histories and no per-sample graph updates. Build it once and reuse
    it for training and evaluation; it is not a per-prefix causal graph.

    Histories are filtered for padding before taking adjacent transitions and
    suffixes. Edges are binary, undirected, and deduplicated across users.
    Shared contains the union of both domain graphs and the Cartesian product
    of the last ``cross_window`` items in each history (co-user association,
    not chronological cross-domain transitions). Zero disables cross edges.
    All three graphs have unit self loops at 1..N-1 before symmetric degree
    normalization, including isolated and opposite-domain nodes. Padding has
    no edges. Storage is O(N + E), with no dense adjacency allocation.
    """
    _integer(num_items, "num_items", 1)
    _integer(cross_window, "cross_window", 0)
    if (not isinstance(target_item_ids, Tensor) or target_item_ids.ndim != 1
            or target_item_ids.layout != torch.strided
            or target_item_ids.dtype not in _INTEGER_DTYPES):
        raise ValueError("target_item_ids must be a one-dimensional integer tensor")
    targets = set(target_item_ids.detach().cpu().tolist())
    if any(item <= 0 or item >= num_items for item in targets):
        raise ValueError("target_item_ids must be in [1, num_items)")

    def validate_item(item: object, target: bool, padding: bool = True) -> int:
        _integer(item, "item ID", 0)
        if item >= num_items:
            raise ValueError("item ID must be smaller than num_items")
        if item == 0:
            if padding:
                return item
            raise ValueError("positive_target_item cannot be padding")
        if (item in targets) != target:
            raise ValueError("item ID belongs to the wrong domain")
        return item

    latest: dict[int | str, tuple[int | float, list[int], list[int]]] = {}
    with train_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be a JSON object")
                user = row["user_id"]
                if type(user) not in (int, str):
                    raise ValueError("user_id must be an integer or string")
                timestamp = row["timestamp"]
                if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
                    raise ValueError("timestamp must be a finite number")
                histories = []
                for field, is_target in (("source_items", False), ("target_items", True)):
                    if not isinstance(row[field], list):
                        raise ValueError(f"{field} must be a list")
                    history = [validate_item(item, is_target) for item in row[field]]
                    histories.append([item for item in history if item != 0])
                validate_item(row["positive_target_item"], True, padding=False)
                if user not in latest or timestamp >= latest[user][0]:
                    latest[user] = (timestamp, histories[0], histories[1])
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                raise ValueError(f"{train_path}:{line_number}: {exc}") from exc

    source_edges: set[tuple[int, int]] = set()
    target_edges: set[tuple[int, int]] = set()
    cross_edges: set[tuple[int, int]] = set()

    def connect(edges: set[tuple[int, int]], left: int, right: int) -> None:
        edges.add((left, right))
        edges.add((right, left))

    for _, source, target in latest.values():
        for history, edges in ((source, source_edges), (target, target_edges)):
            for left, right in zip(history, history[1:]):
                connect(edges, left, right)
        if cross_window:
            for left in source[-cross_window:]:
                for right in target[-cross_window:]:
                    connect(cross_edges, left, right)
    return {
        "shared": _normalized_graph(source_edges | target_edges | cross_edges, num_items),
        "source": _normalized_graph(source_edges, num_items),
        "target": _normalized_graph(target_edges, num_items),
    }


class SparseDualGraphEncoder(nn.Module):
    """Propagate supplied text vectors with independent trainable path projections.

    Each output is ``projection(mean(X, A X, ..., A**layers X))``. There are no
    learned item-ID features. Graphs are cast to the input device and dtype;
    normal module conventions require projection parameters to match the input.
    The residual includes row 0, so callers wanting zero padding features must
    supply a zero vector there. Bias-free projections preserve that zero.
    """

    def __init__(self, hidden_dim: int, layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = _integer(hidden_dim, "hidden_dim", 1)
        self.layers = _integer(layers, "layers", 0)
        self.projections = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in _PATHS
        })

    def forward(self, item_vectors: Tensor, graphs: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(shared, source, target)``, each shaped ``[N, hidden_dim]``."""
        if (item_vectors.layout != torch.strided or item_vectors.ndim != 2
                or item_vectors.shape[1] != self.hidden_dim
                or not item_vectors.is_floating_point()):
            raise ValueError("item_vectors must be a floating [N, hidden_dim] tensor")
        outputs = []
        for name in _PATHS:
            graph = graphs.get(name)
            if (not isinstance(graph, Tensor) or graph.layout != torch.sparse_coo
                    or graph.sparse_dim() != 2 or graph.dense_dim() != 0
                    or graph.shape != (item_vectors.shape[0], item_vectors.shape[0])):
                raise ValueError(f"graphs[{name!r}] must be a sparse COO [N, N] tensor")
            graph = graph.to(device=item_vectors.device, dtype=item_vectors.dtype).coalesce()
            state = item_vectors
            total = item_vectors
            for _ in range(self.layers):
                state = torch.sparse.mm(graph, state)
                total = total + state
            outputs.append(self.projections[name](total / (self.layers + 1)))
        return outputs[0], outputs[1], outputs[2]
