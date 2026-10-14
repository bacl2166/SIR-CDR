"""Finite parameter-only refinement using existing LLM validation scores.

No new recommender, training objective or metric. All four primary metrics are
protected relative to the original runtime setting; test is separately locked.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cdr_framework import llm_training
from cdr_framework.formal_training import _sha256,_write_json,load_fixed_catalog
from cdr_framework.ops import calibrate_scores
from cdr_framework.text_training import load_rows
from experiments.run_llm_cdr import absolute_config
from experiments.tune_llm_fusion import collect_scores,score_metrics,validate_checkpoint

PRIMARY=("R@5","R@10","N@5","N@10")


def parameter_settings(config):
    return [{"pool":config.rerank_candidates,"weight":config.generation_score_weight,
             "normalization":config.fusion_normalization}] + [
        {"pool":p,"weight":w,"normalization":"none"}
        for p in (500,1000) for w in (.1,.5,1.,2.,4.)] + [
        {"pool":p,"weight":w,"normalization":"zscore"}
        for p in (500,1000) for w in (.375,.625,.75)]


def rank_parameters(cache,setting,k):
    pool=min(setting["pool"],cache["candidates"].shape[1])
    scores=cache["retrieval"][:,:pool]
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise ValueError("Nonfinite retrieval score; only negative infinity masks are allowed")
    eligible=torch.isfinite(scores)
    if setting["weight"]>0:
        gen=cache["generation"][:,:pool]
        if not torch.isfinite(gen[eligible]).all():
            raise ValueError("Nonfinite generation score on eligible candidate")
        scores=calibrate_scores(scores,eligible,setting["normalization"]) + setting["weight"]*calibrate_scores(
            gen,eligible,setting["normalization"])
    scores=scores.masked_fill(~eligible,-torch.inf)
    values,order=scores.topk(min(k,pool),dim=1)
    ranked=cache["candidates"][:,:pool].gather(1,order).masked_fill(~torch.isfinite(values),0)
    return torch.nn.functional.pad(ranked,(0,max(0,k-ranked.shape[1])))


def select_protected(results):
    anchor=results[0]["metrics"]
    qualified=[r for r in results if all(r["metrics"][key]>=anchor[key]-1e-12 for key in PRIMARY)]
    return max(qualified,key=lambda r:(r["metrics"]["N@10"],r["metrics"]["R@10"],
                                      r["metrics"]["N@5"],r["metrics"]["R@5"],r==results[0],
                                      -r["setting"]["pool"],-r["setting"]["weight"]))


def audit_ranks(reference,direct,cache,fresh,rows,cutoffs):
    before=score_metrics(cache,reference,cutoffs);after=score_metrics(fresh,direct,cutoffs)
    differences={key:after[key]-before[key] for key in before if key.startswith(("R@","N@"))}
    changed=[]
    for n,row in enumerate(rows):
        a,b=reference[n].tolist(),direct[n].tolist();positive=row.positive_target_item
        ra=a.index(positive)+1 if positive in a else None
        rb=b.index(positive)+1 if positive in b else None
        if ra!=rb:
            detail=[]
            for item in sorted(set(a+b)-{0}):
                scores={"item_id":item}
                for name,data in (("cached",cache),("direct",fresh)):
                    pos=(data["candidates"][n]==item).nonzero()
                    scores[name]=None if not len(pos) else {
                        "candidate_position":int(pos[0,0])+1,
                        "retrieval":float(data["retrieval"][n,int(pos[0,0])]),
                        "generation":float(data["generation"][n,int(pos[0,0])])}
                detail.append(scores)
            changed.append({"row":n,"user_id":row.user_id,"positive":positive,
                            "cached_rank":ra,"direct_rank":rb,"cached_topk":a,"direct_topk":b,
                            "raw_score_details":detail})
    return {"cached_metrics":before,"direct_metrics":after,"metric_differences":differences,
            "ranking_mismatch_rows":int((reference!=direct).any(1).sum()),
            "positive_rank_changes":changed,
            "verified_all_metrics":all(abs(v)<1e-12 for v in differences.values())}


def run(config,cache_root,output,action,device):
    if config.retrieval_score_weight!=1.:
        raise ValueError("This finite protocol fixes retrieval_score_weight=1")
    test_path=config.output_dir/"test.json"
    if action!="test" and test_path.exists():
        raise RuntimeError("Existing test report/reservation: validation tuning refused")
    checkpoint=config.output_dir/"best_model.pt"
    state,current=validate_checkpoint(config,checkpoint)
    phase1=json.loads((cache_root/"search.json").read_text(encoding="utf-8"))
    stored=torch.load(cache_root/"validation_scores.pt",map_location="cpu",weights_only=True)
    identity=stored["identity"]
    if not stored["complete"] or identity!=phase1["identity"] or identity["split"]!="validation":
        raise RuntimeError("Incomplete or inconsistent validation cache")
    if identity["checkpoint_sha256"]!=_sha256(checkpoint):
        raise RuntimeError("Cache checkpoint mismatch")
    for key in ("version","config","artifacts","code","libraries"):
        if identity["training_identity"].get(key)!=current[key]:
            raise RuntimeError(f"Cache {key} mismatch")
    if identity["tool_sha256"]!=_sha256(ROOT/"experiments/tune_llm_fusion.py") or identity["scoring_config"]!=llm_training.config_payload(config):
        raise RuntimeError("Use the unchanged scoring tool/configuration")
    cache=stored["scores"];catalog=load_fixed_catalog(config)
    if cache["candidates"].shape[1]!=min(1000,len(catalog.target_item_ids)):
        raise RuntimeError("Cache candidate pool size differs")
    rows=load_rows(config,catalog,"validation")
    if not torch.equal(cache["positive"],torch.tensor([r.positive_target_item for r in rows])):
        raise RuntimeError("Validation/cache sample order differs")
    settings=parameter_settings(config)
    search_identity={"schema":"llm-parameters-v1","checkpoint_sha256":_sha256(checkpoint),
                     "phase1_report_sha256":_sha256(cache_root/"search.json"),
                     "cache_sha256":_sha256(cache_root/"validation_scores.pt"),"training_identity":current,
                     "tool_sha256":_sha256(Path(__file__)),"settings":settings,"split":"validation",
                     "cumulative_search_budget":38,"protected_metrics":list(PRIMARY)}
    search_identity=json.loads(json.dumps(search_identity))
    output.mkdir(parents=True,exist_ok=True);report_path=output/"search.json"
    if report_path.exists() and json.loads(report_path.read_text())["identity"]!=search_identity:
        raise RuntimeError("Search identity changed; choose a new tuning directory")
    if action=="sweep":
        results=[]
        for setting in settings:
            ranked=rank_parameters(cache,setting,max(config.top_ks))
            metrics=score_metrics(cache,ranked,config.top_ks)
            results.append({"setting":setting,"metrics":metrics})
            print(json.dumps(results[-1]),flush=True)
        best=select_protected(results)
        for result in results:
            result["all_primary_nondecreasing"]=all(result["metrics"][k]>=results[0]["metrics"][k]-1e-12 for k in PRIMARY)
        selected=replace(config,rerank_candidates=best["setting"]["pool"],
                         generation_score_weight=best["setting"]["weight"],
                         fusion_normalization=best["setting"]["normalization"],inference_mode="hybrid")
        (output/"selected.yaml").write_text(yaml.safe_dump(
            {"llm_recommendation":llm_training.config_payload(selected)},sort_keys=False),encoding="utf-8")
        improved=any(best["metrics"][k]>results[0]["metrics"][k]+1e-12 for k in PRIMARY)
        _write_json(report_path,{"identity":search_identity,"complete":True,"results":results,"anchor":results[0],
                                 "best":best,"improved_without_primary_drop":improved,
                                 "selection_policy":"all four >= anchor, then N@10/R@10/N@5/R@5"})
        print(json.dumps({"best":best,"improved_without_primary_drop":improved}),flush=True)
        return
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if report["identity"]!=search_identity or not report["complete"]:
        raise RuntimeError("Complete matching parameter search required")
    setting=report["best"]["setting"]
    report_hash=_sha256(report_path)
    if action=="verify":
        model,_=llm_training.build_model(config,device)
        llm_training.restore_state(model,state["model"])
        fresh=collect_scores(model,rows,config,device,1000,
                             progress=lambda _,n:print(f"Verify: {n}/{len(rows)}",flush=True))
        audit=audit_ranks(rank_parameters(cache,setting,max(config.top_ks)),
                          rank_parameters(fresh,setting,max(config.top_ks)),cache,fresh,rows,config.top_ks)
        old_path=config.output_dir/"validation_hybrid.json"
        verification={"identity":search_identity,"selected_setting":setting,"search_report_sha256":report_hash,
                      "audit":audit,"original_direct_validation":json.loads(old_path.read_text()) if old_path.exists() else None,
                      "verified_all_metrics":audit["verified_all_metrics"]}
        _write_json(output/"verification.json",verification)
        print(json.dumps({"verified_all_metrics":audit["verified_all_metrics"],
                          "positive_rank_changes":len(audit["positive_rank_changes"]),
                          "metric_differences":audit["metric_differences"]}),flush=True)
        if not audit["verified_all_metrics"]:
            raise RuntimeError("Metric mismatch; inspect saved per-row audit before test")
        return
    verify_path=output/"verification.json"
    verification=json.loads(verify_path.read_text(encoding="utf-8"))
    if (verification["identity"]!=search_identity or not verification["verified_all_metrics"]
            or verification["selected_setting"]!=setting or verification["search_report_sha256"]!=report_hash):
        raise RuntimeError("Selected setting/report differs from verified setting")
    test_identity={"schema":"llm-parameter-test-v1","search_identity":search_identity,
                   "selected_setting":setting,"search_report_sha256":report_hash,
                   "verification_sha256":_sha256(verify_path),"split":"test"}
    if test_path.exists():
        saved=json.loads(test_path.read_text())
        if saved.get("parameter_identity")==test_identity and "metrics" in saved:
            print(json.dumps(saved),flush=True);return
        raise RuntimeError("Existing test result/reservation differs or is unfinished; refusing overwrite")
    reservation={"state":"reserved","pid":os.getpid(),"parameter_identity":test_identity}
    # The same test.json is reserved before loading test labels, so a later
    # legacy evaluator rejects this identity too. Never run two test processes.
    with test_path.open('x',encoding='utf-8') as stream:
        json.dump(reservation,stream)
    lock_path=config.output_dir/"test.lock"
    try:
        with lock_path.open('x',encoding='utf-8') as stream:
            json.dump(reservation,stream)
        model,_=llm_training.build_model(config,device)
        llm_training.restore_state(model,state["model"])
        test_rows=load_rows(config,catalog,"test")
        start=time.perf_counter()
        scores=collect_scores(model,test_rows,config,device,1000,
                              progress=lambda _,n:print(f"Test: {n}/{len(test_rows)}",flush=True))
        ranked=rank_parameters(scores,setting,max(config.top_ks))
        metrics=score_metrics(scores,ranked,config.top_ks)
        metrics.update(seconds=time.perf_counter()-start,
                       coverage=len(set(ranked.flatten().tolist())-{0})/len(catalog.target_item_ids),
                       positives_in_seen=sum(r.positive_target_item in r.seen_items for r in test_rows))
        result={"epoch":state["epoch"],"split":"test","mode":"hybrid","checkpoint_sha256":_sha256(checkpoint),
                "metrics":metrics,"parameter_identity":test_identity,"state":"complete"}
        if json.loads(test_path.read_text())!=reservation:
            raise RuntimeError("Test reservation changed during evaluation")
        _write_json(test_path,result);_write_json(lock_path,{"state":"complete","parameter_identity":test_identity})
        print(json.dumps(result),flush=True)
    except BaseException as error:
        if json.loads(test_path.read_text())==reservation:
            _write_json(test_path,{**reservation,"state":"failed","error":repr(error)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--cache-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--action",choices=("sweep","verify","test"),default="sweep")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args();config=absolute_config(args.config)
    if not {5,10}.issubset(config.top_ks):parser.error("This protocol requires cutoffs 5 and 10")
    run(config,args.cache_root.resolve(),args.output.resolve(),args.action,torch.device(args.device))


if __name__=="__main__":main()
