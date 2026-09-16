from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
)
from cdr_framework.metrics import hit_rate_at_k, ndcg_at_k, mrr_at_k


def training_popularity(path: Path) -> Counter:
    # Sliding prefixes repeat events. Count each user's latest training prefix once.
    latest = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["timestamp"], len(row["target_items"]))
            user = row["user_id"]
            if user not in latest or key > latest[user][0]:
                latest[user] = (key, row)
    counts = Counter()
    for _, row in latest.values():
        counts.update(row["target_items"])
        counts.update([row["positive_target_item"]])
    return counts


def summarize(rankings, positives, cutoffs):
    result = {}
    for k in cutoffs:
        result[f"HR@{k}"] = hit_rate_at_k(rankings, positives, k)
        result[f"NDCG@{k}"] = ndcg_at_k(rankings, positives, k)
        result[f"MRR@{k}"] = mrr_at_k(rankings, positives, k)
    return result


@torch.no_grad()
def diagnose(config, device):
    checkpoint = torch.load(config.output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Checkpoint schema mismatch.")
    if checkpoint.get("config") != _config_payload(config):
        raise RuntimeError("Checkpoint configuration mismatch; use the original training YAML.")
    if checkpoint.get("artifacts") != _artifact_fingerprints(config):
        raise RuntimeError("Input artifacts changed since training.")
    catalog = load_fixed_catalog(config)
    model = SIRCDRRecommender(
        catalog, hidden_dim=config.hidden_dim, reasoning_steps=config.reasoning_steps,
        prefix_length=config.prefix_length, retrieval_temperature=config.retrieval_temperature,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    rows = load_recommendation_rows(
        config.processed_dir / "validation.jsonl", max_sequence_length=config.max_sequence_length,
    )
    if not rows:
        raise ValueError("Validation split is empty.")
    counts = training_popularity(config.processed_dir / "train.jsonl")
    popular = sorted(catalog.target_item_ids.tolist(), key=lambda item: (-counts[item], item))
    loader = DataLoader(rows, batch_size=config.evaluation_batch_size,
                        collate_fn=collate_recommendation_rows, shuffle=False)
    retrieval_rows, reranked_rows, popular_rows, positives = [], [], [], []
    candidate_count = min(max(max(config.top_ks), config.rerank_candidates), len(popular))
    max_k = max(config.top_ks)
    for index, batch in enumerate(loader):
        batch = batch.to(device)
        retrieved = model.rank_full_catalog(
            batch, top_k=candidate_count, generation_weight=0.0,
            candidate_chunk_size=config.candidate_chunk_size,
            rerank_candidates=config.rerank_candidates,
        ).cpu().tolist()
        reranked = model.rank_full_catalog(
            batch, top_k=max_k, candidate_chunk_size=config.candidate_chunk_size,
            rerank_candidates=config.rerank_candidates,
        ).cpu().tolist()
        for offset, (history, length) in enumerate(zip(batch.target_items.cpu(), batch.target_lengths.cpu())):
            seen = set(history[:int(length)].tolist())
            # Remove any -inf placeholders when fewer than K eligible items exist.
            retrieval_rows.append([item for item in retrieved[offset] if item not in seen])
            reranked_rows.append([item for item in reranked[offset] if item not in seen])
            popular_rows.append([item for item in popular if item not in seen][:max_k])
        positives.extend(batch.positive_items.cpu().tolist())
        if (index + 1) % 10 == 0 or index + 1 == len(loader):
            print(f"Validation batches: {index + 1}/{len(loader)}", flush=True)
    return {
        "split": "validation", "best_epoch": checkpoint["epoch"],
        "examples": len(rows), "target_catalog_size": len(popular),
        "candidate_count": candidate_count,
        "candidate_recall": hit_rate_at_k(retrieval_rows, positives, candidate_count),
        "retrieval_only": summarize(retrieval_rows, positives, config.top_ks),
        "generation_reranked": summarize(reranked_rows, positives, config.top_ks),
        "training_popularity": summarize(popular_rows, positives, config.top_ks),
        "mask_policy": "exclude target history truncated to max_sequence_length, matching current evaluator",
        "positives_in_masked_history": sum(row.positive_target_item in row.target_items for row in rows),
        "artifact_sha256": checkpoint["artifacts"],
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose retrieval and reranking on validation only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = RecommendationTrainingConfig.from_yaml(args.config)
    config = replace(config, **{
        name: path if path.is_absolute() else ROOT / path
        for name in ("processed_dir", "tokenizer_dir", "output_dir")
        for path in [getattr(config, name)]
    })
    result = diagnose(config, torch.device(args.device))
    output = config.output_dir / "validation_diagnostics.json"
    temporary = output.with_suffix(".json.part")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Diagnostics: {output}")


if __name__ == "__main__":
    main()
