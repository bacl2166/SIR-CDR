"""Validation-only cached fusion search for compact LLM checkpoints; no training."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdr_framework import llm_training
from cdr_framework.formal_training import _sha256, _write_json, _save_atomic
from cdr_framework.ops import calibrate_scores
from cdr_framework.text_data import collate_text_rows
from cdr_framework.text_training import load_rows
from experiments.run_llm_cdr import absolute_config


def sweep_settings():
    return [(100, 0.)] + [(n,w) for n in (100,300,500,1000) for w in (.1,.25,.5,1.,2.)]


@torch.no_grad()
def collect_scores(model, rows, config, device, maximum, existing=None, progress=None):
    model.eval()
    items = model.item_states()
    vectors = F.normalize(items["base"], dim=-1)
    k = min(len(model.target_ids), maximum)
    cache = existing or {name: [] for name in ("candidates", "retrieval", "generation", "positive")}
    start = len(cache["positive"])
    arrays = {name: [torch.as_tensor(cache[name])] if start else [] for name in cache}
    loader = DataLoader(rows[start:], batch_size=config.evaluation_batch_size, collate_fn=collate_text_rows)
    completed = start
    for batch, seen in loader:
        batch = batch.to(device)
        state = model.encode(batch, items)
        scores = state["query"] @ vectors[model.target_ids].T / config.retrieval_temperature
        for i, history in enumerate(seen):
            index = model.target_index[torch.tensor(sorted(history), device=device, dtype=torch.long)]
            scores[i, index[index >= 0]] = -torch.inf
        values, indices = scores.topk(k, dim=1)
        candidates = model.target_ids[indices]
        flat = candidates.flatten()
        batch_index = torch.arange(len(candidates), device=device).repeat_interleave(k)
        generation = torch.empty(len(flat), device=device)
        for offset in range(0, len(flat), config.decode_chunk_size):
            end = offset + config.decode_chunk_size
            generation[offset:end] = model.sequence_scores(state["prefix"][batch_index[offset:end]], flat[offset:end])
        for name, value in (("candidates",candidates),("retrieval",values),
                            ("generation",generation.reshape_as(values)),("positive",batch.positive_items)):
            arrays[name].append(value.cpu())
        completed += len(candidates)
        if progress and (completed % 100 < len(candidates) or completed == len(rows)):
            current = {name: torch.cat(parts) for name,parts in arrays.items()}
            progress(current, completed)
    return {name: torch.cat(parts) for name,parts in arrays.items()}


def cached_rank(cache, candidates, weight, top_k):
    available = cache["candidates"].shape[1]
    if candidates > available or candidates < 1:
        raise ValueError("Requested pool is not available")
    values = cache["retrieval"][:, :candidates]
    eligible = torch.isfinite(values)
    if weight:
        gen = cache["generation"][:, :candidates]
        values = calibrate_scores(values,eligible,"zscore") + weight * calibrate_scores(gen,eligible,"zscore")
    values = values.masked_fill(~eligible,-torch.inf)
    scores, order = values.topk(min(top_k,candidates), dim=1)
    ranked = cache["candidates"][:, :candidates].gather(1,order).masked_fill(~torch.isfinite(scores),0)
    return F.pad(ranked,(0,max(0,top_k-ranked.shape[1])))


def score_metrics(cache, ranked, cutoffs):
    positive = cache["positive"].reshape(-1,1)
    matches = ranked == positive
    result = {"examples": len(ranked)}
    for k in cutoffs:
        hits = matches[:,:k]
        discounts = 1 / torch.log2(torch.arange(2,hits.shape[1]+2,dtype=torch.float64))
        recall = hits.any(1).double().mean().item()
        ndcg = (hits * discounts).sum(1).mean().item()
        result.update({f"HR@{k}": recall,f"R@{k}":recall,f"NDCG@{k}":ndcg,f"N@{k}":ndcg})
    return result


def validate_checkpoint(config, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    identity = llm_training.signature(config)
    for key in ("version","config","artifacts","code","libraries"):
        if state["identity"].get(key) != identity[key]:
            raise RuntimeError(f"Checkpoint {key} mismatch")
    return state, identity


def run(config, device, output, action):
    if (config.output_dir / "test.json").exists():
        raise RuntimeError("Test was already evaluated; this confirmatory search refuses test-informed tuning")
    checkpoint = config.output_dir / "best_model.pt"
    state, training_identity = validate_checkpoint(config, checkpoint)
    # A finite, predeclared search. Max pool is clipped only to the true catalog.
    maximum = 1000
    identity = {"schema":"llm-fusion-cache-v1", "split":"validation",
                "checkpoint_sha256":_sha256(checkpoint), "training_identity":training_identity,
                "tool_sha256":_sha256(Path(__file__)), "maximum":maximum,
                "scoring_config":llm_training.config_payload(config), "settings":sweep_settings()}
    # Convert tuples to their persisted JSON representation for exact reuse.
    identity = json.loads(json.dumps(identity))
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "validation_scores.pt"
    report_path = output / "search.json"
    if report_path.exists() and json.loads(report_path.read_text())["identity"] != identity:
        raise RuntimeError("Search identity changed; use a new tuning directory")
    cache = None
    complete = False
    if cache_path.exists():
        stored = torch.load(cache_path, map_location="cpu", weights_only=True)
        if stored["identity"] != identity:
            raise RuntimeError("Cache identity mismatch")
        cache = stored["scores"]
        complete = stored["complete"]
    if action in ("cache","all") and not complete:
        model, catalog = llm_training.build_model(config, device)
        llm_training.restore_state(model,state["model"])
        rows = load_rows(config,catalog,"validation")
        if cache is not None and len(cache["positive"]) > len(rows):
            raise RuntimeError("Cache has too many rows")
        def progress(scores, completed):
            _save_atomic(cache_path,{"identity":identity,"scores":scores,"complete":completed==len(rows)})
            print(f"Validation cache: {completed}/{len(rows)} ({completed/len(rows):.1%})",flush=True)
        if cache is None or len(cache["positive"]) != len(rows):
            cache = collect_scores(model,rows,config,device,maximum,cache,progress)
            progress(cache,len(rows))
        del model, catalog
        if device.type == "cuda": torch.cuda.empty_cache()
        if action == "cache": return
    elif action == "cache":
        print("Reusing complete validation cache",flush=True)
        return
    if cache is None or not torch.load(cache_path,map_location="cpu",weights_only=True)["complete"]:
        raise RuntimeError("Complete validation cache required")
    if action in ("sweep","all"):
        actual_pool = cache["candidates"].shape[1]
        results=[]
        for count,weight in sweep_settings():
            count=min(count,actual_pool)
            metrics=score_metrics(cache,cached_rank(cache,count,weight,max(config.top_ks)),config.top_ks)
            eligible=torch.isfinite(cache["retrieval"][:,:count])
            recall=((cache["candidates"][:,:count]==cache["positive"][:,None]) & eligible).any(1).double().mean().item()
            results.append({"rerank_candidates":count,"generation_score_weight":weight,
                            "candidate_recall":recall,**metrics})
        best=max(results,key=lambda r:(r["NDCG@10"],-r["generation_score_weight"],-r["rerank_candidates"]))
        selected=replace(config,generation_score_weight=best["generation_score_weight"],
                         retrieval_score_weight=1.,rerank_candidates=best["rerank_candidates"],
                         fusion_normalization="zscore",
                         inference_mode="retrieval" if best["generation_score_weight"]==0 else "hybrid")
        selected_path=output / "selected.yaml"
        payload={"llm_recommendation":llm_training.config_payload(selected)}
        if selected_path.exists() and yaml.safe_load(selected_path.read_text()) != payload:
            raise RuntimeError("Selected configuration already differs")
        selected_path.write_text(yaml.safe_dump(payload,sort_keys=False),encoding="utf-8")
        _write_json(report_path,{"identity":identity,"complete":True,"selection_metric":"NDCG@10",
                                 "results":results,"best":best,"selected_config":str(selected_path)})
        print(json.dumps({"best":best,"selected_config":str(selected_path)}),flush=True)
    if action == "verify":
        report=json.loads(report_path.read_text())
        selected=absolute_config(output / "selected.yaml")
        result=llm_training.evaluate_checkpoint(selected,device,"validation")
        differences={k:abs(result["metrics"][k]-report["best"][k]) for k in result["metrics"] if k.startswith(("R@","N@"))}
        if max(differences.values())>1e-6:
            raise RuntimeError(f"Cached/direct metrics differ: {differences}")
        _write_json(output / "verification.json",{"identity":identity,"verified":True,"direct":result})
        print("Cached ranking agrees with direct validation; freeze selected.yaml before test.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--action",choices=("cache","sweep","all","verify"),default="all")
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    config=absolute_config(args.config)
    if max(config.top_ks)>100 or 10 not in config.top_ks:
        parser.error("Fixed protocol requires NDCG@10 and top_ks <= 100")
    run(config,torch.device(args.device),args.output.resolve(),args.action)


if __name__ == "__main__": main()
