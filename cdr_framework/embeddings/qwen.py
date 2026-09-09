from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from cdr_framework.embeddings.providers import BaseTextEmbeddingProvider


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ItemText:
    item_id: int
    domain: str
    text: str


@dataclass(frozen=True)
class EmbeddingRunResult:
    output_dir: Path
    embeddings_path: Path
    item_count: int
    api_batches: int
    resumed_batches: int


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_item_texts(path: str | Path, limit: int | None = None) -> list[ItemText]:
    items: list[ItemText] = []
    seen: set[int] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            item_id = int(payload["item_id"])
            text = str(payload["text"]).strip()
            domain = str(payload["domain"])
            if item_id <= 0 or item_id in seen or not text or not domain:
                raise ValueError(f"Invalid item text at {path}, line {line_number}.")
            seen.add(item_id)
            items.append(ItemText(item_id, domain, text))
            if limit is not None and len(items) >= limit:
                break
    items.sort(key=lambda item: item.item_id)
    if not items:
        raise ValueError(f"No item texts found in {path}.")
    return items


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_tensor_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def _encode_with_retry(
    provider: BaseTextEmbeddingProvider,
    texts: Sequence[str],
    max_retries: int,
    initial_delay: float,
    sleep: Callable[[float], None],
) -> torch.Tensor:
    for attempt in range(max_retries + 1):
        try:
            return provider.encode_text(texts).detach().cpu().to(torch.float32)
        except Exception:
            if attempt >= max_retries:
                raise
            sleep(initial_delay * (2**attempt))
    raise RuntimeError("unreachable")


def _managed_output(path: Path) -> bool:
    for marker in (path / "progress.json", path / "manifest.json"):
        if marker.exists():
            try:
                if json.loads(marker.read_text(encoding="utf-8")).get("schema_version") == SCHEMA_VERSION:
                    return True
            except (OSError, ValueError):
                pass
    return False


def run_embedding_job(
    items: Sequence[ItemText],
    provider: BaseTextEmbeddingProvider,
    output_dir: str | Path,
    *,
    input_sha256: str,
    model: str,
    dimension: int,
    batch_size: int,
    max_retries: int,
    retry_initial_seconds: float,
    force: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> EmbeddingRunResult:
    output = Path(output_dir)
    if force and output.exists():
        if not _managed_output(output):
            raise RuntimeError(f"Refusing to delete unmanaged output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "model": model,
        "dimension": dimension,
        "batch_size": batch_size,
        "item_count": len(items),
    }
    progress_path = output / "progress.json"
    if progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if previous.get(key) != value:
                raise RuntimeError(f"Embedding resume mismatch for {key}; use --force to restart.")

    api_batches = 0
    resumed_batches = 0
    batch_paths: list[Path] = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(items), batch_size)):
        batch = items[start : start + batch_size]
        path = chunks / f"batch-{batch_index:06d}.pt"
        batch_paths.append(path)
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["item_ids"].tolist() != [item.item_id for item in batch]:
                raise RuntimeError(f"Cached batch item mismatch: {path}")
            if tuple(payload["embeddings"].shape) != (len(batch), dimension):
                raise RuntimeError(f"Cached batch shape mismatch: {path}")
            resumed_batches += 1
        else:
            vectors = _encode_with_retry(
                provider,
                [item.text for item in batch],
                max_retries,
                retry_initial_seconds,
                sleep,
            )
            if tuple(vectors.shape) != (len(batch), dimension):
                raise RuntimeError(
                    f"Embedding batch shape {tuple(vectors.shape)} does not match "
                    f"{(len(batch), dimension)}."
                )
            _save_tensor_atomic(
                path,
                {
                    "item_ids": torch.tensor([item.item_id for item in batch], dtype=torch.long),
                    "embeddings": vectors,
                },
            )
            api_batches += 1
        _write_json_atomic(
            progress_path,
            {**identity, "completed_batches": batch_index + 1, "total_batches": total_batches},
        )

    max_item_id = max(item.item_id for item in items)
    matrix = torch.zeros((max_item_id + 1, dimension), dtype=torch.float32)
    for path in batch_paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        matrix[payload["item_ids"]] = payload["embeddings"].to(torch.float32)
    embeddings_path = output / "text_embeddings.pt"
    _save_tensor_atomic(embeddings_path, matrix)
    _write_json_atomic(
        output / "item_index.json",
        {
            "padding_item_id": 0,
            "row_is_item_id": True,
            "item_ids": [item.item_id for item in items],
        },
    )
    _write_json_atomic(
        output / "manifest.json",
        {
            **identity,
            "dtype": "float32",
            "matrix_shape": list(matrix.shape),
            "padding_row_is_zero": True,
            "completed_batches": total_batches,
        },
    )
    return EmbeddingRunResult(output, embeddings_path, len(items), api_batches, resumed_batches)
