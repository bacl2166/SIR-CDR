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

from cdr_framework.experiment_config import TokenizerTrainingConfig
from cdr_framework.tokenization.residual_kmeans import fit_residual_kmeans
from cdr_framework.tokenization.tokenizer import DomainAdaptiveSemanticTokenizer


SCHEMA_VERSION = 2
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
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = DomainAdaptiveSemanticTokenizer(input_dim, hidden_dim, num_domains=2)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.LayerNorm(input_dim))
        self.domain_classifier = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        vectors: torch.Tensor,
        domains: torch.Tensor,
        domain_loss_weight: float,
        gate_balance_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encoder(vectors, domains)
        reconstruction = self.decoder(encoded.final)
        reconstruction_loss = F.mse_loss(reconstruction, vectors.float())
        domain_loss = F.cross_entropy(
            self.domain_classifier(encoded.domain_specific), domains.long()
        )
        gate_balance = (encoded.gate.mean() - 0.5).pow(2)
        total = (
            reconstruction_loss
            + domain_loss_weight * domain_loss
            + gate_balance_weight * gate_balance
        )
        return encoded.final, {
            "total": total,
            "reconstruction": reconstruction_loss,
            "domain": domain_loss,
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
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                "Tokenizer output uses an obsolete quantizer schema; rerun with --force."
            )
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
    model = TrainableSemanticTokenizer(config.input_dim, config.hidden_dim).to(target_device)
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
        if state.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Tokenizer checkpoint schema mismatch; use --force to restart.")
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
        totals = {
            "total": 0.0,
            "reconstruction": 0.0,
            "domain": 0.0,
            "gate_balance": 0.0,
        }
        batches = 0
        for vectors, domains in loader:
            vectors = vectors.to(target_device, non_blocking=True)
            domains = domains.to(target_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, losses = model(
                vectors,
                domains,
                config.domain_loss_weight,
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
            latent_rows[item_ids] = encoded.final.cpu().float()

    item_latents = latent_rows[dataset.item_ids].to(target_device)
    item_tokens_device, quantized_latents, codebooks = fit_residual_kmeans(
        item_latents,
        codebook_size=config.codebook_size,
        token_length=config.token_length,
        iterations=config.kmeans_iterations,
        seed=config.seed,
    )
    item_tokens = item_tokens_device.cpu()
    token_rows[dataset.item_ids] = item_tokens
    unique_sequences = len({tuple(row) for row in item_tokens.tolist()})
    collision_rate = 1.0 - unique_sequences / len(dataset.item_ids)
    per_level_utilization = [
        torch.unique(item_tokens[:, level]).numel() / config.codebook_size
        for level in range(config.token_length)
    ]
    codebook_utilization = sum(per_level_utilization) / len(per_level_utilization)
    quantization_mse = float(F.mse_loss(quantized_latents, item_latents).cpu())
    _save_atomic(output / "semantic_ids.pt", token_rows)
    _save_atomic(output / "item_latents.pt", latent_rows)
    _save_atomic(
        output / "tokenizer.pt",
        {
            "schema_version": SCHEMA_VERSION,
            "model_state": model.state_dict(),
            "codebooks": codebooks.cpu(),
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
    quality_report = {
        "collision_rate": collision_rate,
        "max_collision_rate": config.max_collision_rate,
        "per_level_codebook_utilization": per_level_utilization,
        "min_level_utilization": config.min_level_utilization,
        "quantization_mse": quantization_mse,
        "passed": collision_rate <= config.max_collision_rate
        and min(per_level_utilization) >= config.min_level_utilization,
    }
    _write_json(output / "quality_report.json", quality_report)
    if not quality_report["passed"]:
        raise RuntimeError(
            "Semantic ID quality gate failed: "
            f"collision_rate={collision_rate:.6f}, "
            f"per_level_utilization={per_level_utilization}. "
            "Inspect quality_report.json before retrying."
        )
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
        "per_level_codebook_utilization": per_level_utilization,
        "quantization_mse": quantization_mse,
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
