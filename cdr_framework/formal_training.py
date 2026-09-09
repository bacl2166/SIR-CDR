from __future__ import annotations

import json
import hashlib
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from cdr_framework.experiment_config import RecommendationTrainingConfig
from cdr_framework.formal_recommendation import (
    FixedCatalog,
    SIRCDRRecommender,
    collate_recommendation_rows,
    load_recommendation_rows,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FormalTrainingResult:
    output_dir: Path
    epochs_completed: int
    best_validation: dict[str, float]
    already_complete: bool = False


def _config_payload(config: RecommendationTrainingConfig) -> dict:
    payload = asdict(config)
    for name in ("processed_dir", "tokenizer_dir", "output_dir"):
        payload[name] = str(payload[name])
    payload["top_ks"] = list(payload["top_ks"])
    return payload


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


def _artifact_fingerprints(config: RecommendationTrainingConfig) -> dict[str, str]:
    paths = {
        "mappings": config.processed_dir / "mappings.json",
        "train": config.processed_dir / "train.jsonl",
        "validation": config.processed_dir / "validation.jsonl",
        "test": config.processed_dir / "test.jsonl",
        "item_latents": config.tokenizer_dir / "item_latents.pt",
        "semantic_ids": config.tokenizer_dir / "semantic_ids.pt",
        "tokenizer": config.tokenizer_dir / "tokenizer.pt",
        "tokenizer_quality": config.tokenizer_dir / "quality_report.json",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def load_fixed_catalog(config: RecommendationTrainingConfig) -> FixedCatalog:
    quality = json.loads(
        (config.tokenizer_dir / "quality_report.json").read_text(encoding="utf-8")
    )
    if quality.get("passed") is not True:
        raise RuntimeError("Formal training requires a passed tokenizer quality report.")
    tokenizer_checkpoint = torch.load(
        config.tokenizer_dir / "tokenizer.pt", map_location="cpu", weights_only=True
    )
    if tokenizer_checkpoint.get("schema_version") != 2:
        raise RuntimeError("Formal training requires tokenizer schema_version=2.")
    mappings = json.loads(
        (config.processed_dir / "mappings.json").read_text(encoding="utf-8")
    )
    target_domain = mappings["target_domain"]
    target_item_ids = torch.tensor(
        [
            item_id
            for item_id, item in enumerate(mappings["id_to_item"])
            if item is not None and item["domain"] == target_domain
        ],
        dtype=torch.long,
    )
    return FixedCatalog(
        item_latents=torch.load(
            config.tokenizer_dir / "item_latents.pt",
            map_location="cpu",
            weights_only=True,
        ),
        semantic_ids=torch.load(
            config.tokenizer_dir / "semantic_ids.pt",
            map_location="cpu",
            weights_only=True,
        ),
        codebooks=tokenizer_checkpoint["codebooks"],
        target_item_ids=target_item_ids,
    )


@torch.no_grad()
def evaluate_full_catalog(
    model: SIRCDRRecommender,
    rows: list,
    config: RecommendationTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(
        rows,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        collate_fn=collate_recommendation_rows,
    )
    cutoffs = sorted(set(config.top_ks))
    maximum = max(cutoffs)
    totals = {f"HR@{k}": 0.0 for k in cutoffs}
    totals.update({f"NDCG@{k}": 0.0 for k in cutoffs})
    totals.update({f"MRR@{k}": 0.0 for k in cutoffs})
    recommended = set()
    count = 0
    for batch in loader:
        batch = batch.to(device)
        ranked = model.rank_full_catalog(
            batch,
            top_k=maximum,
            rerank_candidates=config.rerank_candidates,
            candidate_chunk_size=config.candidate_chunk_size,
        )
        positives = batch.positive_items
        recommended.update(ranked.detach().cpu().reshape(-1).tolist())
        count += len(positives)
        for row in range(len(positives)):
            matches = (ranked[row] == positives[row]).nonzero(as_tuple=False)
            rank = int(matches[0, 0]) + 1 if len(matches) else None
            for k in cutoffs:
                if rank is not None and rank <= k:
                    totals[f"HR@{k}"] += 1.0
                    totals[f"NDCG@{k}"] += 1.0 / math.log2(rank + 1)
                    totals[f"MRR@{k}"] += 1.0 / rank
    metrics = {name: value / max(count, 1) for name, value in totals.items()}
    metrics["Coverage"] = len(recommended) / len(model.target_item_ids)
    metrics["Examples"] = float(count)
    return metrics


def run_formal_training(
    config: RecommendationTrainingConfig,
    *,
    device: str | torch.device | None = None,
    force: bool = False,
    report: Callable[[dict], None] | None = None,
) -> FormalTrainingResult:
    output = config.output_dir
    manifest_path = output / "manifest.json"
    config_payload = _config_payload(config)
    artifacts = _artifact_fingerprints(config)
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != config_payload:
            raise RuntimeError("Completed recommendation configuration changed; use --force.")
        if manifest.get("artifacts") != artifacts:
            raise RuntimeError("Recommendation input artifacts changed; use --force.")
        return FormalTrainingResult(
            output,
            int(manifest["epochs_completed"]),
            manifest["best_validation"],
            True,
        )
    if force and output.exists():
        if not (output / "last_checkpoint.pt").exists() and not manifest_path.exists():
            raise RuntimeError(f"Refusing to delete unmanaged directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    catalog = load_fixed_catalog(config)
    model = SIRCDRRecommender(
        catalog,
        hidden_dim=config.hidden_dim,
        reasoning_steps=config.reasoning_steps,
        prefix_length=config.prefix_length,
        retrieval_temperature=config.retrieval_temperature,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_rows = load_recommendation_rows(
        config.processed_dir / "train.jsonl",
        max_sequence_length=config.max_sequence_length,
    )
    validation_rows = load_recommendation_rows(
        config.processed_dir / "validation.jsonl",
        max_sequence_length=config.max_sequence_length,
    )
    history = []
    start_epoch = 0
    best_score = -1.0
    best_validation: dict[str, float] = {}
    stale_evaluations = 0
    checkpoint_path = output / "last_checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
        if checkpoint.get("config") != config_payload:
            raise RuntimeError("Recommendation resume configuration mismatch; use --force.")
        if checkpoint.get("artifacts") != artifacts:
            raise RuntimeError("Recommendation input artifacts changed; use --force.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        history = checkpoint["history"]
        best_score = float(checkpoint["best_score"])
        best_validation = checkpoint["best_validation"]
        stale_evaluations = int(checkpoint["stale_evaluations"])

    epochs_completed = start_epoch
    for epoch in range(start_epoch, config.max_epochs):
        loader = DataLoader(
            train_rows,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + epoch),
            collate_fn=collate_recommendation_rows,
        )
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for batch in loader:
            batch = batch.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch)
            result.losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in result.losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
        record = {
            "epoch": epoch + 1,
            **{name: value / batches for name, value in totals.items()},
        }
        should_evaluate = (epoch + 1) % config.evaluation_every == 0 or epoch + 1 == config.max_epochs
        if should_evaluate:
            validation = evaluate_full_catalog(model, validation_rows, config, target_device)
            record["validation"] = validation
            score = validation[f"NDCG@{max(config.top_ks)}"]
            if score > best_score:
                best_score = score
                best_validation = validation
                stale_evaluations = 0
                _save_atomic(
                    output / "best_model.pt",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "config": _config_payload(config),
                        "artifacts": artifacts,
                        "validation": validation,
                    },
                )
            else:
                stale_evaluations += 1
        history.append(record)
        epochs_completed = epoch + 1
        _save_atomic(
            checkpoint_path,
            {
                "schema_version": SCHEMA_VERSION,
                "epoch": epochs_completed,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_score": best_score,
                "best_validation": best_validation,
                "stale_evaluations": stale_evaluations,
                "config": _config_payload(config),
                "artifacts": artifacts,
            },
        )
        _write_json(output / "training_history.json", history)
        if report:
            report(record)
        if should_evaluate and stale_evaluations >= config.patience:
            break

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "epochs_completed": epochs_completed,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "target_catalog_size": len(catalog.target_item_ids),
        "full_target_evaluation": True,
        "random_negative_sampling": False,
        "best_validation": best_validation,
        "config": config_payload,
        "artifacts": artifacts,
    }
    _write_json(manifest_path, manifest)
    return FormalTrainingResult(output, epochs_completed, best_validation)


def evaluate_saved_model(
    config: RecommendationTrainingConfig,
    *,
    split: str = "test",
    device: str | torch.device | None = None,
) -> dict[str, float]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test.")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    catalog = load_fixed_catalog(config)
    model = SIRCDRRecommender(
        catalog,
        hidden_dim=config.hidden_dim,
        reasoning_steps=config.reasoning_steps,
        prefix_length=config.prefix_length,
        retrieval_temperature=config.retrieval_temperature,
    ).to(target_device)
    checkpoint = torch.load(
        config.output_dir / "best_model.pt", map_location=target_device, weights_only=True
    )
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Recommendation checkpoint schema mismatch.")
    if checkpoint.get("config") != _config_payload(config):
        raise RuntimeError("Recommendation checkpoint configuration mismatch.")
    if checkpoint.get("artifacts") != _artifact_fingerprints(config):
        raise RuntimeError("Recommendation input artifacts changed since training.")
    model.load_state_dict(checkpoint["model_state"])
    rows = load_recommendation_rows(
        config.processed_dir / f"{split}.jsonl",
        max_sequence_length=config.max_sequence_length,
    )
    return evaluate_full_catalog(model, rows, config, target_device)
