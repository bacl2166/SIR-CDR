from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdr_framework.experiment_config import RecommendationTrainingConfig
from cdr_framework.formal_recommendation import (
    SIRCDRRecommender, collate_recommendation_rows, load_recommendation_rows,
)
from cdr_framework.formal_training import (
    SCHEMA_VERSION, _artifact_fingerprints, _config_payload, load_fixed_catalog,
    _sha256, _write_json,
)
from cdr_framework.metrics import hit_rate_at_k
from experiments.diagnose_recommender import summarize


def best_result(results):
    # Fixed selection criterion; prefer smaller pools/weights when metrics tie.
    return max(results, key=lambda row: (
        row["NDCG@10"], -row["candidates"], -row["generation_weight"],
    ))


@torch.no_grad()
def sweep(config, device, candidates=(200, 500, 1000), weights=(0.0, 0.5, 1.0, 2.0), batch_size=16):
    candidates = sorted(set(candidates))
    weights = sorted(set(weights))
    if not candidates or any(k < 20 for k in candidates):
        raise ValueError("Candidate counts must be at least 20.")
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights) or batch_size < 1:
        raise ValueError("Weights must be finite/nonnegative and batch size positive.")
    checkpoint_path = config.output_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    artifacts = _artifact_fingerprints(config)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Checkpoint schema mismatch.")
    if checkpoint.get("config") != _config_payload(config):
        raise RuntimeError("Use the unchanged training YAML; sweep settings are separate CLI arguments.")
    if checkpoint.get("artifacts") != artifacts:
        raise RuntimeError("Input artifacts changed since training.")
    catalog = load_fixed_catalog(config)
    if max(candidates) > len(catalog.target_item_ids):
        raise ValueError("Requested candidate count exceeds target catalog size.")
    model = SIRCDRRecommender(
        catalog, hidden_dim=config.hidden_dim, reasoning_steps=config.reasoning_steps,
        prefix_length=config.prefix_length, retrieval_temperature=config.retrieval_temperature,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rows = load_recommendation_rows(config.processed_dir / "validation.jsonl",
                                    max_sequence_length=config.max_sequence_length)
    if not rows:
        raise ValueError("Validation split is empty.")
    identity = {
        "config": _config_payload(config), "artifacts": artifacts,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "ranking_code_sha256": _sha256(ROOT / "cdr_framework/formal_recommendation.py"),
        "script_sha256": _sha256(Path(__file__)),
        "candidates": candidates, "weights": weights, "batch_size": batch_size,
        "selection_metric": "NDCG@10", "split": "validation",
        "best_epoch": checkpoint["epoch"], "torch_version": str(torch.__version__),
        "device": str(device),
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    output = config.output_dir / f"validation_sweep_{run_id}.json"
    report = {"identity": identity, "results": [], "complete": False}
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["identity"] != identity:
            raise RuntimeError("Sweep identity mismatch.")
    done = {(r["candidates"], r["generation_weight"]) for r in report["results"]}
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_recommendation_rows)
    for count in candidates:
        for weight in weights:
            if (count, weight) in done:
                print(f"Reusing completed group: candidates={count}, weight={weight}", flush=True)
                continue
            start = time.perf_counter()
            rankings, positives = [], []
            for index, batch in enumerate(loader):
                batch = batch.to(device)
                ranked = model.rank_full_catalog(
                    batch, top_k=count if weight == 0 else 20,
                    rerank_candidates=count, generation_weight=weight,
                    retrieval_weight=1.0, candidate_chunk_size=config.candidate_chunk_size,
                ).cpu().tolist()
                for predicted, history, length in zip(ranked, batch.target_items.cpu(), batch.target_lengths.cpu()):
                    seen = set(history[:int(length)].tolist())
                    rankings.append([item for item in predicted if item not in seen])
                positives.extend(batch.positive_items.cpu().tolist())
                if (index + 1) % 25 == 0 or index + 1 == len(loader):
                    print(f"Pool={count} alpha={weight}: batches {index + 1}/{len(loader)}", flush=True)
            result = {
                "candidates": count, "generation_weight": weight, "retrieval_weight": 1.0,
                "examples": len(rows), "seconds": time.perf_counter() - start,
                **summarize(rankings, positives, (5, 10, 20)),
            }
            if weight == 0:
                result["candidate_recall"] = hit_rate_at_k(rankings, positives, count)
            report["results"].append(result)
            report["best_so_far"] = best_result(report["results"])
            _write_json(output, report)
            print(json.dumps(result, sort_keys=True), flush=True)
    report["complete"] = True
    report["best_validation_setting"] = best_result(report["results"])
    _write_json(output, report)
    print("Best validation setting: " + json.dumps(report["best_validation_setting"], sort_keys=True))
    print(f"Report: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Validation-only candidate and fusion sweep; no retraining.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", nargs="+", type=int, default=[200, 500, 1000])
    parser.add_argument("--weights", nargs="+", type=float, default=[0, 0.5, 1, 2])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = RecommendationTrainingConfig.from_yaml(args.config)
    config = replace(config, **{
        name: path if path.is_absolute() else ROOT / path
        for name in ("processed_dir", "tokenizer_dir", "output_dir")
        for path in [getattr(config, name)]
    })
    # Include retrieval-only groups to measure each candidate pool's recall.
    weights = sorted(set([0.0, *args.weights]))
    sweep(config, torch.device(args.device), args.candidates, weights, args.batch_size)


if __name__ == "__main__":
    main()
