"""Prepare once, smoke-test on the server, then train the opt-in Qwen reasoner."""
from pathlib import Path
import argparse
from dataclasses import replace
import json
import sys
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdr_framework.llm_config import LLMCDRConfig
from cdr_framework import llm_training
from cdr_framework.formal_training import load_fixed_catalog
from cdr_framework.text_data import collate_text_rows
from cdr_framework.text_training import load_rows

MODEL_FILES = ["config.json", "model*.safetensors", "model.safetensors.index.json"]


def absolute_config(path, root=ROOT):
    config = LLMCDRConfig.from_yaml(path)
    return replace(config, **{name: (root / getattr(config, name)).resolve()
        for name in ("processed_dir", "tokenizer_dir", "output_dir")})


def _validate_snapshot(path):
    path = Path(path)
    metadata = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if metadata.get("model_type") != "qwen3":
        raise ValueError("Expected a dense Qwen3 text model")
    index = path / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if not shards:
        raise ValueError("Empty model weight index")
    for shard in shards:
        if Path(shard).name != shard:
            raise ValueError("Invalid model shard path")
        if not (path / shard).is_file():
            raise FileNotFoundError(f"Missing model shard: {path / shard}")
    return path


def check_cached_model(config):
    from huggingface_hub import snapshot_download
    config.require_pinned_model()
    snapshot = snapshot_download(config.llm_model_name, revision=config.llm_revision,
        allow_patterns=MODEL_FILES, local_files_only=True)
    if Path(snapshot).name != config.llm_revision:
        raise ValueError("Cached snapshot revision mismatch")
    return _validate_snapshot(snapshot)


def prepare_model(config, output):
    from huggingface_hub import snapshot_download
    snapshot = _validate_snapshot(snapshot_download(config.llm_model_name,
        revision=config.llm_revision, allow_patterns=MODEL_FILES))
    resolved = replace(config, llm_revision=snapshot.name, llm_local_files_only=True)
    resolved.require_pinned_model()
    payload = {"llm_recommendation": llm_training.config_payload(resolved)}
    output = Path(output)
    if output.exists():
        if yaml.safe_load(output.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("Resolved config already exists with different settings; use another path")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".yaml.part")
        temporary.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        temporary.replace(output)
    print(json.dumps({"resolved_config": str(output), "model_revision": snapshot.name,
                      "snapshot": str(snapshot)}), flush=True)
    return resolved


def preflight(config):
    snapshot = check_cached_model(config)
    catalog = load_fixed_catalog(config)
    report = {"model_revision": config.llm_revision, "snapshot": str(snapshot),
              "items": len(catalog.item_latents), "target_items": len(catalog.target_item_ids),
              "rows": {split: len(load_rows(config, catalog, split))
                       for split in ("train", "validation", "test")}}
    if not all(report["rows"].values()):
        raise ValueError("Every data split must be nonempty")
    if catalog.item_latents.shape[1] != config.hidden_dim:
        # Existing codebook summary consumes raw centroids of hidden_dim width.
        raise ValueError("hidden_dim must match cached tokenizer latent/codebook width")
    print(json.dumps(report), flush=True)
    return report


def smoke(config, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(config.seed)
    model, catalog = llm_training.build_model(config, device)
    rows = load_rows(config, catalog, "train")[:config.batch_size]
    batch, seen = collate_text_rows(rows)
    model.train()
    losses = model(batch.to(device))
    losses["total"].backward()
    groups = {
        "projection": [model.reasoner.project.weight],
        "memory": [model.reasoner.memory.write.weight],
        "lora": [p for n, p in model.reasoner.backbone.named_parameters() if "lora_" in n],
    }
    gradients = {name: sum(float(p.grad.float().abs().sum()) for p in params if p.grad is not None)
                 for name, params in groups.items()}
    if not torch.isfinite(losses["total"]) or any(not (0 < x < float("inf")) for x in gradients.values()):
        raise RuntimeError(f"Smoke check found missing/nonfinite gradient: {gradients}")
    if any(p.grad is not None for n, p in model.reasoner.backbone.named_parameters() if "lora_" not in n):
        raise RuntimeError("Frozen backbone unexpectedly received parameter gradients")
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        ranked = model.rank(batch.to(device), seen, max(config.top_ks))
    report = {"loss": float(losses["total"].detach()), "gradient_l1": gradients,
              "ranking_shape": list(ranked.shape), "optimizer_steps": 0,
              "peak_cuda_GiB": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None}
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", required=True, choices=("prepare", "preflight", "smoke", "train", "evaluate"))
    parser.add_argument("--resolved-config", help="Required for prepare; output YAML with immutable model SHA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--mode", choices=("retrieval", "hybrid", "generate", "exhaustive"))
    parser.add_argument("--max-epochs-this-run", type=int)
    args = parser.parse_args()
    if args.max_epochs_this_run is not None and args.max_epochs_this_run < 1:
        parser.error("--max-epochs-this-run must be positive")
    config = absolute_config(args.config)
    if args.action == "prepare":
        if not args.resolved_config:
            parser.error("prepare requires --resolved-config")
        prepare_model(config, args.resolved_config)
        return
    preflight(config)
    if args.action == "preflight":
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.action == "smoke":
        smoke(config, device)
    elif args.action == "train":
        print(json.dumps(llm_training.train(config, device, max_epochs_this_run=args.max_epochs_this_run)), flush=True)
    else:
        print(json.dumps(llm_training.evaluate_checkpoint(config, device, args.split, args.mode)), flush=True)


if __name__ == "__main__":
    main()
