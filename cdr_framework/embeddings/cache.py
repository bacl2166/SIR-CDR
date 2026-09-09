from __future__ import annotations

import re
from pathlib import Path

import torch


class CachedEmbeddingStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def exists(self, namespace: str, key: str) -> bool:
        return self._path(namespace, key).exists()

    def load(self, namespace: str, key: str) -> torch.Tensor:
        path = self._path(namespace, key)
        if not path.exists():
            raise FileNotFoundError(f"Cached embedding not found: {path}")
        return torch.load(path, weights_only=True)

    def save(self, namespace: str, key: str, tensor: torch.Tensor) -> Path:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.detach().cpu(), path)
        return path

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / _safe_name(namespace) / f"{_safe_name(key)}.pt"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return safe.strip("-") or "default"
