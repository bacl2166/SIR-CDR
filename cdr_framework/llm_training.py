"""Isolated LLM runner; no changes to Text v4 checkpoint semantics."""
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import random
import time
import torch
from torch.utils.data import DataLoader

from cdr_framework.formal_training import _artifact_fingerprints, _save_atomic, _sha256, _write_json, load_fixed_catalog
from cdr_framework.llm_efficiency import DeviceLossMeter
from cdr_framework.llm_model import LLMSIRCDR
from cdr_framework.llm_reasoner import QwenLatentReasoner
from cdr_framework.text_data import collate_text_rows
from cdr_framework.text_graph import build_training_graphs
from cdr_framework import text_training

VERSION = "llm-cdr-v1"


def config_payload(config):
    payload = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def signature(config):
    config.require_pinned_model()
    payload = config_payload(config)
    policy = {key: payload.pop(key) for key in text_training.RUNTIME_RANKING_FIELDS}
    files = ["llm_config.py", "llm_reasoner.py", "llm_training.py", "llm_model.py", "llm_efficiency.py", "text_config.py",
             "text_data.py", "text_graph.py", "text_model.py", "text_training.py",
             "modules.py", "modules_cpf.py", "ops.py", "losses.py", "catalog_generation.py",
             "formal_training.py", "formal_recommendation.py", "experiment_config.py"]
    return {"version": VERSION, "config": payload, "selection_policy": policy,
            "artifacts": _artifact_fingerprints(config),
            "code": {name: _sha256(Path(__file__).parent / name) for name in files},
            "libraries": {name: version(name) for name in ("torch", "transformers", "peft")}}


def build_model(config, device):
    config.require_pinned_model()
    if device.type == "cpu" and config.llm_dtype != "float32":
        raise ValueError("CPU runs require llm_dtype=float32")
    # Keep Text v4's factory and fingerprint untouched. The LLM-only subclass
    # adds an evaluation cache, with the same catalog validation as that factory.
    catalog = load_fixed_catalog(config)
    if not torch.isfinite(catalog.item_latents).all() or not torch.isfinite(catalog.codebooks).all():
        raise ValueError("Non-finite tokenizer artifact")
    if (catalog.semantic_ids[1:] < 0).any() or (catalog.semantic_ids[1:] >= catalog.codebooks.shape[1]).any():
        raise ValueError("Invalid semantic token")
    if catalog.item_latents.shape[1] != catalog.codebooks.shape[2]:
        raise ValueError("Latent/codebook dimensions differ")
    graphs = build_training_graphs(config.processed_dir / "train.jsonl", len(catalog.item_latents),
                                   catalog.target_item_ids, cross_window=config.cross_window)
    model = LLMSIRCDR(catalog, graphs, config).to(device)
    model.reasoner = QwenLatentReasoner.from_pretrained(config).to(device)
    return model, catalog


@torch.no_grad()
def evaluate(model, rows, config, device, mode=None):
    model.eval()
    started = time.perf_counter()
    with model.evaluation_catalog():
        result = text_training.evaluate(model, rows, config, device, mode)
    # Include the one-time catalog build in reported evaluation wall time.
    result['seconds'] = time.perf_counter() - started
    return result


def _compact_keys(model):
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    return {key for key in model.state_dict()
            if not key.startswith("reasoner.backbone.") or key in trainable}


def compact_state(model):
    keys = _compact_keys(model)
    return {key: tensor.detach().cpu() for key, tensor in model.state_dict().items() if key in keys}


def restore_state(model, state):
    expected = _compact_keys(model)
    if set(state) != expected:
        raise RuntimeError(f"Compact checkpoint keys mismatch: missing={sorted(expected - set(state))}, "
                           f"unexpected={sorted(set(state) - expected)}")
    # Only frozen base tensors may be absent; expected was derived before load.
    model.load_state_dict(state, strict=False)


def train(config, device, *, max_epochs_this_run=None):
    identity = signature(config)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        result = json.loads(manifest.read_text(encoding="utf-8"))
        if result["identity"] != identity or not (output / "best_model.pt").exists():
            raise RuntimeError("Completed run identity changed or checkpoint missing; choose a new output")
        return result
    state_path = output / "last_checkpoint.pt"
    if not state_path.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty output without resumable checkpoint; choose a new output")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model, catalog = build_model(config, device)
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5,
        patience=config.scheduler_patience, min_lr=1e-6)
    rows = text_training.load_rows(config, catalog, "train")
    validation = text_training.load_rows(config, catalog, "validation")
    if not rows or not validation:
        raise ValueError("Train and validation sets must be nonempty")
    history, start_epoch, stale, best_score, best_validation = [], 0, 0, -1., {}
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise RuntimeError("Resume identity mismatch; choose a new output")
        restore_state(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, start_epoch, stale = state["history"], state["epoch"], state["stale"]
        best_score, best_validation = state["best_score"], state["best_validation"]
        torch.set_rng_state(state["rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    print(json.dumps({"version": VERSION, "revision": config.llm_revision,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "total_parameters": sum(p.numel() for p in model.parameters())}), flush=True)
    epochs_done = start_epoch
    for epoch in range(start_epoch, config.max_epochs):
        if stale >= config.patience:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loader = DataLoader(rows, batch_size=config.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + epoch), collate_fn=collate_text_rows)
        meter, group_examples, updates = DeviceLossMeter(), 0, 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        train_started = time.perf_counter()
        for batch_number, (batch, _) in enumerate(loader, 1):
            losses = model(batch.to(device))
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Non-finite LLM training loss")
            count = len(batch.positive_items)
            (losses["total"] * count).backward()
            group_examples += count
            meter.add(losses, count)
            if batch_number % config.gradient_accumulation_steps == 0 or batch_number == len(loader):
                for param in parameters:
                    if param.grad is not None:
                        param.grad.div_(group_examples)
                torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                group_examples = 0
                updates += 1
                if updates == 1 or updates % 20 == 0:
                    loss_value = float(losses['total'].detach())
                    rate = meter.examples / (time.perf_counter() - train_started)
                    print(f"epoch={epoch+1} batch={batch_number}/{len(loader)} updates={updates} "
                          f"samples={meter.examples}/{len(rows)} loss={loss_value:.6f} "
                          f"samples_per_second={rate:.3f}", flush=True)
        means = meter.means()
        train_seconds = time.perf_counter() - train_started
        record = {"epoch": epoch + 1, "optimizer_steps": updates,
                  "lr": optimizer.param_groups[0]["lr"],
                  **means, "train_seconds": train_seconds,
                  "train_samples_per_second": len(rows) / train_seconds,
                  "train_peak_cuda_GiB": (torch.cuda.max_memory_allocated(device) / 2**30
                                          if device.type == "cuda" else None)}
        if epoch == 0 or (epoch + 1) % config.evaluation_every == 0 or epoch + 1 == config.max_epochs:
            metrics = evaluate(model, validation, config, device)
            record["validation"] = metrics
            scheduler.step(metrics[config.selection_metric])
            if metrics[config.selection_metric] > best_score:
                stale, best_score, best_validation = 0, metrics[config.selection_metric], metrics
                _save_atomic(output / "best_model.pt", {"identity": identity, "epoch": epoch + 1,
                    "model": compact_state(model), "validation": metrics})
            else:
                stale += 1
        history.append(record)
        epochs_done = epoch + 1
        _save_atomic(state_path, {"identity": identity, "epoch": epochs_done,
            "model": compact_state(model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history, "stale": stale,
            "best_score": best_score, "best_validation": best_validation,
            "rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        _write_json(output / "history.json", history)
        print(json.dumps(record), flush=True)
        if max_epochs_this_run is not None and epochs_done - start_epoch >= max_epochs_this_run:
            break
    result = {"identity": identity, "epochs_completed": epochs_done,
              "best_validation": best_validation,
              "complete": epochs_done >= config.max_epochs or stale >= config.patience,
              "checkpoint_format": "non-backbone parameters and LoRA; pinned base loaded separately"}
    if result["complete"]:
        _write_json(manifest, result)
    return result


def evaluate_checkpoint(config, device, split="validation", mode=None):
    if split not in {"validation", "test"}:
        raise ValueError("Only validation or test evaluation is allowed")
    checkpoint_path = config.output_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    current = signature(config)
    for key in ("version", "config", "artifacts", "code", "libraries"):
        if checkpoint["identity"].get(key) != current[key]:
            raise RuntimeError(f"Evaluation {key} mismatch")
    mode = mode or config.inference_mode
    evaluation = text_training.evaluation_identity(config, checkpoint_path, split, mode)
    # One test report across all ranking modes; changing mode cannot bypass the lock.
    path = config.output_dir / ("test.json" if split == "test" else f"validation_{mode}.json")
    if path.exists() and split == "test":
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("evaluation_identity") != evaluation:
            raise RuntimeError("Existing test evaluation identity differs; test result is immutable")
        return existing
    model, catalog = build_model(config, device)
    restore_state(model, checkpoint["model"])
    rows = text_training.load_rows(config, catalog, split)
    if not rows:
        raise ValueError("Evaluation set is empty")
    result = {"epoch": checkpoint["epoch"], "split": split, "mode": mode,
              "evaluation_identity": evaluation, "checkpoint_sha256": _sha256(checkpoint_path),
              "metrics": evaluate(model, rows, config, device, mode)}
    _write_json(path, result)
    return result
