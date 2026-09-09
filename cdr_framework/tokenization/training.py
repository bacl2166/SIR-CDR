from __future__ import annotations

import json
import hashlib
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cdr_framework.codebook import SemanticCodebook
from cdr_framework.experiment_config import TokenizerTrainingConfig
from cdr_framework.tokenization.tokenizer import DomainAdaptiveSemanticTokenizer


SCHEMA_VERSION = 1
SOURCE_DOMAIN = "Sports_and_Outdoors"
TARGET_DOMAIN = "Clothing_Shoes_and_Jewelry"


@dataclass(frozen=True)
class TokenizerDataset:
    item_ids: torch.Tensor
    vectors: torch.Tensor
    domains: torch.Tensor
    domain_names: tuple[str, ...]


@dataclass(frozen=True)
class TokenizerRunResult:
    output_dir: Path
    semantic_ids_path: Path
    item_count: int
    epochs_completed: int
    collision_rate: float
    codebook_utilization: float
    already_complete: bool = False


class TrainableSemanticTokenizer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, codebook_size: int, seed: int):
        super().__init__()
        self.encoder = DomainAdaptiveSemanticTokenizer(input_dim, hidden_dim, num_domains=2)
        self.codebook = SemanticCodebook.random_init(codebook_size, hidden_dim, seed)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.LayerNorm(input_dim))

    def forward(
        self,
        vectors: torch.Tensor,
        domains: torch.Tensor,
        token_length: int,
        commitment_weight: float,
        gate_balance_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encoder(vectors, domains)
        tokens, quantized, vq_loss = self.codebook.quantize(encoded.final, token_length)
        straight_through = encoded.final + (quantized - encoded.final).detach()
        reconstruction = self.decoder(straight_through)
        reconstruction_loss = F.mse_loss(reconstruction, vectors.float())
        gate_balance = (encoded.gate.mean() - 0.5).pow(2)
        total = (
            reconstruction_loss
            + commitment_weight * vq_loss
            + gate_balance_weight * gate_balance
        )
        return tokens, {
            "total": total,
            "reconstruction": reconstruction_loss,
            "vq": vq_loss,
            "gate_balance": gate_balance,
        }


def load_tokenizer_dataset(
    embeddings_path: str | Path,
    item_texts_path: str | Path,
) -> TokenizerDataset:
    vectors = torch.load(embeddings_path, map_location="cpu", weights_only=True).float()
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("Embedding matrix must have shape [num_items + 1, dimension].")
    if not torch.isfinite(vectors).all():
        raise ValueError("Embedding matrix contains non-finite values.")
    if not torch.equal(vectors[0], torch.zeros_like(vectors[0])):
        raise ValueError("Embedding row 0 must be the padding vector.")

    records: dict[int, str] = {}
    with Path(item_texts_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            item_id = int(payload["item_id"])
            domain = str(payload["domain"])
            if domain not in {SOURCE_DOMAIN, TARGET_DOMAIN}:
                raise ValueError(f"Unknown domain at line {line_number}: {domain}")
            if item_id in records:
                raise ValueError(f"Duplicate item_id at line {line_number}: {item_id}")
            records[item_id] = domain
    expected = set(range(1, vectors.shape[0]))
    if set(records) != expected:
        raise ValueError("Item text IDs do not exactly match embedding matrix rows.")

    item_ids = torch.arange(1, vectors.shape[0], dtype=torch.long)
    domain_names = tuple(records[item_id] for item_id in item_ids.tolist())
    domains = torch.tensor(
        [0 if domain == SOURCE_DOMAIN else 1 for domain in domain_names],
        dtype=torch.long,
    )
    return TokenizerDataset(item_ids, vectors[1:], domains, domain_names)


def _save_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _managed_output(path: Path) -> bool:
    marker = path / "training_state.pt"
    manifest = path / "manifest.json"
    return marker.exists() or manifest.exists()


def _config_payload(config: TokenizerTrainingConfig) -> dict:
    payload = asdict(config)
    for name in ("embeddings_path", "item_texts_path", "output_dir"):
        payload[name] = str(payload[name])
    return payload


def _checkpoint(
    model: TrainableSemanticTokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    config: TokenizerTrainingConfig,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "config": _config_payload(config),
    }


def train_semantic_tokenizer(
    dataset: TokenizerDataset,
    config: TokenizerTrainingConfig,
    *,
    device: str | torch.device | None = None,
    force: bool = False,
    report: Callable[[dict[str, float]], None] | None = None,
) -> TokenizerRunResult:
    output = Path(config.output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return TokenizerRunResult(
            output,
            output / "semantic_ids.pt",
            manifest["item_count"],
            manifest["epochs_completed"],
            manifest["collision_rate"],
            manifest["codebook_utilization"],
            True,
        )
    if force and output.exists():
        if not _managed_output(output):
            raise RuntimeError(f"Refusing to delete unmanaged tokenizer directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TrainableSemanticTokenizer(
        config.input_dim, config.hidden_dim, config.codebook_size, config.seed
    ).to(target_device)
    if dataset.vectors.shape[1] != config.input_dim:
        raise ValueError(
            f"Embedding dimension {dataset.vectors.shape[1]} does not match "
            f"tokenizer input_dim {config.input_dim}."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float]] = []
    start_epoch = 0
    state_path = output / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=target_device, weights_only=True)
        if state.get("config") != _config_payload(config):
            raise RuntimeError("Tokenizer resume configuration mismatch; use --force to restart.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        history = state["history"]
        start_epoch = int(state["epoch"])

    for epoch in range(start_epoch, config.max_epochs):
        loader = DataLoader(
            TensorDataset(dataset.vectors, dataset.domains),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + epoch),
        )
        model.train()
        totals = {"total": 0.0, "reconstruction": 0.0, "vq": 0.0, "gate_balance": 0.0}
        batches = 0
        for vectors, domains in loader:
            vectors = vectors.to(target_device, non_blocking=True)
            domains = domains.to(target_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, losses = model(
                vectors,
                domains,
                config.token_length,
                config.commitment_weight,
                config.gate_balance_weight,
            )
            losses["total"].backward()
            optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach().cpu())
            batches += 1
        metrics = {"epoch": float(epoch + 1), **{name: value / batches for name, value in totals.items()}}
        history.append(metrics)
        _save_atomic(
            state_path,
            _checkpoint(model, optimizer, epoch + 1, history, config),
        )
        _write_json(output / "training_history.json", history)
        if report:
            report(metrics)

    model.eval()
    token_rows = torch.full(
        (int(dataset.item_ids.max()) + 1, config.token_length), -1, dtype=torch.long
    )
    latent_rows = torch.zeros(
        (int(dataset.item_ids.max()) + 1, config.hidden_dim), dtype=torch.float32
    )
    inference_loader = DataLoader(
        TensorDataset(dataset.item_ids, dataset.vectors, dataset.domains),
        batch_size=config.batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for item_ids, vectors, domains in inference_loader:
            vectors = vectors.to(target_device)
            domains = domains.to(target_device)
            encoded = model.encoder(vectors, domains)
            tokens, _, _ = model.codebook.quantize(encoded.final, config.token_length)
            token_rows[item_ids] = tokens.cpu()
            latent_rows[item_ids] = encoded.final.cpu().float()

    item_tokens = token_rows[dataset.item_ids]
    unique_sequences = len({tuple(row) for row in item_tokens.tolist()})
    collision_rate = 1.0 - unique_sequences / len(dataset.item_ids)
    used_codes = torch.unique(item_tokens).numel()
    codebook_utilization = used_codes / config.codebook_size
    _save_atomic(output / "semantic_ids.pt", token_rows)
    _save_atomic(output / "item_latents.pt", latent_rows)
    _save_atomic(
        output / "tokenizer.pt",
        {
            "schema_version": SCHEMA_VERSION,
            "model_state": model.state_dict(),
            "config": _config_payload(config),
        },
    )
    item_tokens_path = output / "item_semantic_ids.jsonl"
    temporary_item_tokens = item_tokens_path.with_suffix(".jsonl.part")
    with temporary_item_tokens.open("w", encoding="utf-8", newline="\n") as stream:
        for item_id, domain, tokens in zip(
            dataset.item_ids.tolist(), dataset.domain_names, item_tokens.tolist()
        ):
            stream.write(
                json.dumps(
                    {"item_id": item_id, "domain": domain, "tokens": tokens},
                    sort_keys=True,
                )
                + "\n"
            )
    temporary_item_tokens.replace(item_tokens_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "item_count": len(dataset.item_ids),
        "input_dim": config.input_dim,
        "hidden_dim": config.hidden_dim,
        "codebook_size": config.codebook_size,
        "token_length": config.token_length,
        "epochs_completed": config.max_epochs,
        "collision_rate": collision_rate,
        "unique_semantic_ids": unique_sequences,
        "codebook_utilization": codebook_utilization,
        "semantic_ids_shape": list(token_rows.shape),
        "embeddings_sha256": _sha256(config.embeddings_path),
        "item_texts_sha256": _sha256(config.item_texts_path),
    }
    _write_json(manifest_path, manifest)
    return TokenizerRunResult(
        output,
        output / "semantic_ids.pt",
        len(dataset.item_ids),
        config.max_epochs,
        collision_rate,
        codebook_utilization,
    )
