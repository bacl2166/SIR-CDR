"""Paired, validation-only candidate recall diagnostic for frozen Sports HPO jobs."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cdr_framework.catalog_generation import constrained_generate
from cdr_framework.formal_training import _sha256, _write_json
from cdr_framework.ops import calibrate_scores
from cdr_framework.text_data import collate_text_rows
from cdr_framework.text_training import load_rows
from experiments.llm_snapshot_scoring import cpu_item_snapshot, map_tensors, stable_topk
from experiments.recover_llm_evaluation import _identity, _model, fixed_execution, operation_lock
from experiments.run_llm_cdr import absolute_config, check_cached_model
from experiments.tune_llm_fusion import score_metrics, validate_checkpoint
from experiments.tune_sports_clothing import check_training, gpu_snapshot, job_from_key, study_path

CANDIDATE_BUDGET = 500
GENERATION_BUDGET = 50
BEAM_SIZE = 50
CUTOFFS = (5, 10)
ARMS = ("retrieval", "dual")


def merge_candidates(retrieved, generated):
    """Reserve generated slots, then fill with unseen, unique retrieved items."""
    if (retrieved.ndim != 2 or generated.ndim != 2 or
            retrieved.shape[0] != generated.shape[0] or generated.shape[1] > retrieved.shape[1]):
        raise ValueError("Candidate lists must be batched and generation must fit the budget")
    budget = retrieved.shape[1]
    pools = []
    for retrieval_row, generation_row in zip(retrieved.tolist(), generated.tolist()):
        chosen = dict.fromkeys(item for item in generation_row if item)
        for item in retrieval_row:
            if len(chosen) == budget:
                break
            if item:
                chosen.setdefault(item, None)
        # Keep existing retrieval order when generation adds no new items.
        shared = dict.fromkeys(item for item in retrieval_row if item in chosen)
        selected = [*shared, *(item for item in chosen if item not in shared)]
        pools.append(selected + [0] * (budget - len(selected)))
    return retrieved.new_tensor(pools)


def summarize(positives, pools, rankings):
    matched = {}
    results = {}
    for arm in ARMS:
        matched[arm] = pools[arm].eq(positives[:, None]).any(1)
        recall = matched[arm].double().mean().item()
        metrics = score_metrics({"positive": positives}, rankings[arm], CUTOFFS)
        results[arm] = dict(metrics, candidate_recall=recall,
                            mean_candidates=pools[arm].ne(0).sum(1).double().mean().item(),
                            top10_given_recalled=metrics["HR@10"] / recall if recall else None)
    before, after = (matched[arm] for arm in ARMS)
    results["positive_candidate_changes"] = {
        "added": (after & ~before).sum().item(),
        "lost": (before & ~after).sum().item(),
    }
    results["dual_minus_retrieval"] = {
        key: results["dual"][key] - results["retrieval"][key]
        for key in ("candidate_recall", "NDCG@5", "NDCG@10", "HR@5", "HR@10")
    }
    return results


def rerank(model, config, prefix, scores, candidates):
    eligible = candidates.ne(0)
    positions = model.target_index[candidates].clamp_min(0)
    retrieval = scores.gather(1, positions).masked_fill(~eligible, -torch.inf)
    rows, columns = eligible.nonzero(as_tuple=True)
    generation = torch.zeros_like(retrieval)
    for start in range(0, len(rows), config.decode_chunk_size):
        r = rows[start:start + config.decode_chunk_size]
        c = columns[start:start + config.decode_chunk_size]
        generation[r, c] = model.sequence_scores(prefix[r], candidates[r, c])
    if not torch.isfinite(retrieval[eligible]).all() or not torch.isfinite(generation[eligible]).all():
        raise RuntimeError("Invalid score for an eligible candidate")
    combined = config.retrieval_score_weight * calibrate_scores(
        retrieval, eligible, config.fusion_normalization
    ) + config.generation_score_weight * calibrate_scores(
        generation, eligible, config.fusion_normalization
    )
    return stable_topk(combined.masked_fill(~eligible, -torch.inf), candidates, max(CUTOFFS))


@torch.no_grad()
def evaluate_pair(model, rows, config, device, snapshot):
    def timestamp():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return perf_counter()

    items = map_tensors(snapshot, lambda value: value.to(device))
    vectors = F.normalize(items["base"], dim=-1)
    targets = model.target_ids
    budget = min(CANDIDATE_BUDGET, len(targets))
    pools = {arm: [] for arm in ARMS}
    rankings = {arm: [] for arm in ARMS}
    positives = []
    seen_positive = generated_new = generated_total = 0
    common_time = retrieval_time = dual_time = 0.0
    loader = DataLoader(rows, batch_size=config.evaluation_batch_size, collate_fn=collate_text_rows)
    for number, (batch, seen) in enumerate(loader, start=1):
        started = timestamp()
        labels = batch.positive_items
        batch = replace(batch, positive_items=torch.zeros_like(labels)).to(device)
        state = model.encode(batch, items)
        prefix, query = state["prefix"], state["query"]
        scores = query @ vectors[targets].T / config.retrieval_temperature
        for row, history in enumerate(seen):
            indices = model.target_index[torch.tensor(sorted(history), dtype=torch.long, device=device)]
            scores[row, indices[indices >= 0]] = -torch.inf
        retrieved = stable_topk(scores, targets, budget)
        common_time += timestamp() - started

        started = timestamp()
        retrieval_rank = rerank(model, config, prefix, scores, retrieved)
        retrieval_time += timestamp() - started

        started = timestamp()
        generated = constrained_generate(
            model.decoder, prefix, model.trie, BEAM_SIZE, min(GENERATION_BUDGET, budget),
            seen, query, vectors,
        )
        dual = merge_candidates(retrieved, generated)
        dual_rank = rerank(model, config, prefix, scores, dual)
        dual_time += timestamp() - started

        for baseline, proposed, emitted in zip(retrieved.tolist(), dual.tolist(), generated.tolist()):
            a, b, g = set(baseline) - {0}, set(proposed) - {0}, set(emitted) - {0}
            if len(a) != len(b):
                raise RuntimeError("Candidate budgets differ")
            generated_new += len(g - a)
            generated_total += len(g)
        seen_positive += sum(label in history for label, history in zip(labels.tolist(), seen))
        positives.append(labels)
        pools["retrieval"].append(retrieved.cpu())
        pools["dual"].append(dual.cpu())
        rankings["retrieval"].append(retrieval_rank.cpu())
        rankings["dual"].append(dual_rank.cpu())
        if number % 25 == 0 or number == len(loader):
            print(f"Validation batches {number}/{len(loader)}", flush=True)

    results = summarize(torch.cat(positives), {arm: torch.cat(pools[arm]) for arm in ARMS},
                        {arm: torch.cat(rankings[arm]) for arm in ARMS})
    results["positives_in_seen"] = seen_positive
    results["generation_candidates"] = {"emitted": generated_total, "outside_retrieval_pool": generated_new}
    results["timing_seconds"] = {"common_encode_and_retrieve": common_time,
                                "retrieval_total": common_time + retrieval_time,
                                "dual_total": common_time + dual_time}
    return results


def run_diagnostic(config, output, device, source):
    checkpoint = config.output_dir / "best_model.pt"
    manifest = config.output_dir / "manifest.json"
    if config.retrieval_score_weight <= 0 or config.generation_score_weight <= 0:
        raise ValueError("Both arms require the same positive retrieval and generation weights")
    with operation_lock(output):
        report_path = output / "report.json"
        if report_path.exists():
            raise FileExistsError(report_path)
        state, identity = validate_checkpoint(config, checkpoint)
        check_training(json.loads(manifest.read_text(encoding="utf-8")), state, identity)
        with fixed_execution(device, config.seed):
            provenance = _identity(config, checkpoint, identity, device)
            model, catalog = _model(config, state, device)
            rows = load_rows(config, catalog, "validation")
            results = evaluate_pair(model, rows, config, device, cpu_item_snapshot(model))
        if (_sha256(checkpoint) != provenance["checkpoint_sha256"] or
                _sha256(manifest) != provenance["completed_manifest_sha256"]):
            raise RuntimeError("Checkpoint or completed manifest changed during evaluation")
        report = {
            "schema": "sports-dual-recall-v1",
            "split": "validation",
            "source": source,
            "checkpoint_epoch": state["epoch"],
            "identity": {key: value for key, value in provenance.items() if key != "evaluation_policy"},
            "policy": {
                "candidate_budget": CANDIDATE_BUDGET, "generation_budget": GENERATION_BUDGET,
                "beam_size": BEAM_SIZE, "cutoffs": list(CUTOFFS),
                "fusion_normalization": config.fusion_normalization,
                "retrieval_weight": config.retrieval_score_weight,
                "generation_weight": config.generation_score_weight,
                "timing": "paired batch compute; excludes model load, snapshot, data collation and metrics",
                "catalog": "checkpoint CPU float32 snapshot; full target retrieval; seen items masked",
                "execution": "float32, deterministic algorithms, math SDPA, TF32 disabled",
                "ties": "higher score first, then lower item ID",
                "code_sha256": _sha256(Path(__file__)),
            },
            "results": results,
        }
        _write_json(report_path, report)
        return report


def study_job(root, study_id, job_key):
    job = job_from_key(job_key)
    if job is None:
        raise ValueError("Expected a Sports/Clothing HPO job such as sports_to_clothing/c06/seed42")
    study = study_path(root, study_id)
    manifest = study / "study.json"
    digest = _sha256(manifest)
    if json.loads((study / "study_hash.json").read_text(encoding="utf-8")) != {"sha256": digest}:
        raise RuntimeError("HPO study manifest hash mismatch")
    runtime = study / "jobs" / job_key / "runtime.yaml"
    entry = json.loads(manifest.read_text(encoding="utf-8"))["configs"][job_key]
    if (Path(entry["path"]) != runtime or _sha256(runtime) != entry["sha256"] or
            json.loads(runtime.read_text(encoding="utf-8")) != entry["payload"]):
        raise RuntimeError("HPO runtime config differs from its recorded identity")
    config = absolute_config(runtime, root=root)
    if config.output_dir != runtime.parent / "training" or config.seed != job["seed"]:
        raise RuntimeError("HPO config points outside the selected training job")
    return config, {"study_id": study_id, "job": job_key,
                    "study_sha256": digest, "runtime_sha256": entry["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Data/lihb/SIR-CDR"))
    parser.add_argument("--study-id", default="sports-clothing-ndcg-v1")
    parser.add_argument("--job", required=True)
    parser.add_argument("--gpu", required=True, help="Physical GPU UUID approved for this run")
    args = parser.parse_args()
    root = args.root.resolve()
    config, source = study_job(root, args.study_id, args.job)
    check_cached_model(config)
    cards, busy = gpu_snapshot()
    card = next((card for card in cards if card["uuid"] == args.gpu), None)
    if (card is None or args.gpu in busy or card["utilization"] > 5 or card["free_mib"] < 12 * 1024):
        raise RuntimeError("Approved GPU is busy, absent, or has under 12 GiB free")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one approved GPU must be visible")
    output = root / "artifacts/diagnostics/sports-dual-recall-v1" / args.study_id / args.job
    report = run_diagnostic(config, output, torch.device("cuda:0"), source)
    print(json.dumps(report["results"], ensure_ascii=False, indent=2))
    print(f"REPORT {output / 'report.json'}")


if __name__ == "__main__":
    main()
