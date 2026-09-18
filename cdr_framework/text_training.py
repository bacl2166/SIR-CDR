from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from cdr_framework.formal_training import load_fixed_catalog, _artifact_fingerprints, _config_payload, _write_json, _save_atomic, _sha256
from cdr_framework.text_config import TextCDRConfig
from cdr_framework.text_data import load_text_rows, collate_text_rows
from cdr_framework.text_graph import build_training_graphs
from cdr_framework.text_model import TextSIRCDR

VERSION = "text-cdr-v3"
VARIANTS = ("full", "no_graph", "no_reasoning", "single_step", "no_feedback", "no_semantic", "no_cpf_loss",
            "target_only", "no_proto", "no_cd_inj", "no_sp_inj", "no_lsep", "no_lsh", "no_csum", "v2_legacy")


def variant_config(config, variant, seed):
    if variant not in VARIANTS:
        raise ValueError("Unknown variant")
    options = {
        "full": {}, "no_graph": {"graph_enabled": False},
        "no_reasoning": {"reasoning_enabled": False, "feedback_steps": 1},
        "single_step": {"reasoning_steps": 1}, "no_feedback": {"feedback_steps": 1},
        "no_semantic": {"semantic_enabled": False}, "no_cpf_loss": {"cpf_weight": 0.0},
        "target_only": {"source_enabled": False, "graph_enabled": False,
                        "prototype_enabled": False, "cd_injector_enabled": False,
                        "sp_injector_enabled": False},
        "no_proto": {"prototype_enabled": False},
        "no_cd_inj": {"cd_injector_enabled": False},
        "no_sp_inj": {"sp_injector_enabled": False},
        "no_lsep": {"lsep_weight": 0.0},
        "no_lsh": {"contrastive_alignment": False},
        "no_csum": {"codebook_summary_enabled": False},
        "v2_legacy": {"prototype_enabled": False, "cd_injector_enabled": False,
                      "sp_injector_enabled": False, "codebook_summary_enabled": False,
                      "contrastive_alignment": False},
    }
    return replace(config, **options[variant], seed=seed,
                   output_dir=config.output_dir / f"{variant}_seed{seed}")


def signature(config):
    files = ["text_config.py", "text_data.py", "text_model.py", "text_graph.py", "catalog_generation.py", "text_training.py",
             "formal_recommendation.py", "modules.py", "modules_cpf.py", "losses.py"]
    return {"version": VERSION, "config": _config_payload(config), "artifacts": _artifact_fingerprints(config),
            "code": {name: _sha256(Path(__file__).parent / name) for name in files}}


def verify_checkpoint_identity(checkpoint, config, require_code=True):
    """Layered checkpoint identity check.

    version/config/artifacts always must match the stored training identity; the
    code fingerprint may drift for inference-only tooling (explicit opt-in, the
    drift is recorded in the report for audit). The data/config layers can never
    be disabled.
    """
    stored = checkpoint.get("identity") if isinstance(checkpoint, dict) else None
    if not isinstance(stored, dict):
        return False, "missing or invalid stored identity"
    current = signature(config)
    for layer in ("version", "config", "artifacts"):
        if current.get(layer) != stored.get(layer):
            return False, f"{layer} mismatch"
    code_drift = current.get("code") != stored.get("code")
    if code_drift and require_code:
        return False, "code mismatch"
    return True, ("code mismatch allowed (inference-only)" if code_drift else "ok")


def build_model(config, device):
    catalog = load_fixed_catalog(config)
    if not torch.isfinite(catalog.item_latents).all() or not torch.isfinite(catalog.codebooks).all():
        raise ValueError("Non-finite tokenizer artifact")
    if (catalog.semantic_ids[1:] < 0).any() or (catalog.semantic_ids[1:] >= catalog.codebooks.shape[1]).any():
        raise ValueError("Invalid semantic token")
    if catalog.item_latents.shape[1] != catalog.codebooks.shape[2]:
        raise ValueError("Latent/codebook dimensions differ")
    graphs = build_training_graphs(config.processed_dir / "train.jsonl", len(catalog.item_latents),
                                   catalog.target_item_ids, cross_window=config.cross_window)
    model = TextSIRCDR(catalog, graphs, config).to(device)
    return model, catalog


def load_rows(config, catalog, split):
    return load_text_rows(config.processed_dir / f"{split}.jsonl", config.max_sequence_length,
                          len(catalog.item_latents), catalog.target_item_ids)


@torch.no_grad()
def evaluate(model, rows, config, device, mode=None):
    model.eval()
    totals = {f"{m}@{k}": 0.0 for m in ("HR", "NDCG", "MRR") for k in config.top_ks}
    coverage = set()
    start = time.perf_counter()
    loader = DataLoader(rows, batch_size=config.evaluation_batch_size, collate_fn=collate_text_rows)
    for batch, seen in loader:
        ranked = model.rank(batch.to(device), seen, max(config.top_ks), mode=mode).cpu().tolist()
        for items, positive in zip(ranked, batch.positive_items.tolist()):
            coverage.update(item for item in items if item > 0)
            rank = items.index(positive) + 1 if positive in items else None
            for k in config.top_ks:
                if rank is not None and rank <= k:
                    totals[f"HR@{k}"] += 1
                    totals[f"NDCG@{k}"] += 1 / math.log2(rank + 1)
                    totals[f"MRR@{k}"] += 1 / rank
    result = {key: value / len(rows) for key, value in totals.items()}
    result.update({f"R@{k}": result[f"HR@{k}"] for k in config.top_ks})
    result.update({f"N@{k}": result[f"NDCG@{k}"] for k in config.top_ks})
    result.update(examples=len(rows), coverage=len(coverage) / len(model.target_ids),
                  seconds=time.perf_counter() - start,
                  positives_in_seen=sum(row.positive_target_item in row.seen_items for row in rows))
    return result


def train(config, device, *, max_epochs_this_run=None):
    identity = signature(config)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "manifest.json"
    if completed.exists():
        result = json.loads(completed.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise RuntimeError("Completed run inputs/config/code changed; choose a new output root")
        if not (output / "best_model.pt").exists():
            raise RuntimeError("Completed run is missing best_model.pt")
        return result
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model, catalog = build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                          patience=config.scheduler_patience, min_lr=1e-6)
    rows, validation = load_rows(config, catalog, "train"), load_rows(config, catalog, "validation")
    state_path = output / "last_checkpoint.pt"
    history, start_epoch, stale, best_score, best_validation = [], 0, 0, -1.0, {}
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise RuntimeError("Resume inputs/config/code mismatch; use a new output root")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, start_epoch, stale = state["history"], state["epoch"], state["stale"]
        best_score, best_validation = state["best_score"], state["best_validation"]
        torch.set_rng_state(state["rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if not state_path.exists() and any(output.iterdir()):
        raise RuntimeError("Output contains files but no resumable checkpoint; choose a new output root")
    epochs_done = start_epoch
    for epoch in range(start_epoch, config.max_epochs):
        if stale >= config.patience:
            break
        model.train()
        loader = DataLoader(rows, batch_size=config.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + epoch), collate_fn=collate_text_rows)
        totals = {}
        for batch, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = model(batch.to(device))
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Non-finite training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(batch.positive_items)
        record = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                  **{key: value / len(rows) for key, value in totals.items()}}
        if epoch == 0 or (epoch + 1) % config.evaluation_every == 0 or epoch + 1 == config.max_epochs:
            metrics = evaluate(model, validation, config, device)
            record["validation"] = metrics
            score = metrics[config.selection_metric]
            scheduler.step(score)
            if score > best_score:
                stale, best_score, best_validation = 0, score, metrics
                _save_atomic(output / "best_model.pt", {"identity": identity, "epoch": epoch + 1,
                    "model": model.state_dict(), "validation": metrics})
            else:
                stale += 1
        history.append(record)
        epochs_done = epoch + 1
        _save_atomic(state_path, {"identity": identity, "epoch": epochs_done, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
            "stale": stale, "best_score": best_score, "best_validation": best_validation,
            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
        _write_json(output / "history.json", history)
        print(json.dumps(record), flush=True)
        if max_epochs_this_run is not None and epochs_done - start_epoch >= max_epochs_this_run:
            break
    result = {"identity": identity, "epochs_completed": epochs_done, "best_validation": best_validation,
              "complete": epochs_done >= config.max_epochs or stale >= config.patience,
              "protocol": "per-user leave-one-out; training-only static graph; full-history seen exclusion",
              "inference_mode": config.inference_mode,
              "label_smoothing": config.label_smoothing}
    if result["complete"]:
        _write_json(completed, result)
    return result


def evaluate_checkpoint(config, device, split="validation", mode=None, code_check=True):
    if split not in {"validation", "test"}:
        raise ValueError("Invalid split")
    checkpoint = torch.load(config.output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    ok, why = verify_checkpoint_identity(checkpoint, config, require_code=code_check)
    if not ok:
        raise RuntimeError(f"Evaluation checkpoint {why}")
    model, catalog = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    result = {"epoch": checkpoint["epoch"], "split": split, "mode": mode or config.inference_mode,
              "checkpoint_sha256": _sha256(config.output_dir / "best_model.pt"),
              "metrics": evaluate(model, load_rows(config, catalog, split), config, device, mode)}
    _write_json(config.output_dir / f"{split}_{mode or config.inference_mode}.json", result)
    return result
