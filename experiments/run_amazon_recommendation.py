"""Wait for sealed new Amazon catalogs, then run 12 independently seeded recommenders."""
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from cdr_framework import llm_training
from cdr_framework.datasets.gencdr_protocol import sha
from cdr_framework.experiment_control import (
    git_identity, host_resources, query_gpu_inventory, validate_batch_id, write_json_atomic,
)
from experiments import amazon_recommendation_policy as policy
from experiments import prepare_amazon_assets as assets
from experiments import run_gencdr_shared as shared
from experiments import run_llm_cdr as llm
from experiments import recover_llm_evaluation as recovery
from experiments.continue_amazon_assets import ResourceUnavailable
from experiments.run_gencdr_six import check_host_pressure, admit_host_budget, run_commands, training_memory_requirement
from experiments.tune_llm_fusion import validate_checkpoint
from experiments.tune_llm_parameters import parameter_settings

PAIRS = ("food_to_kitchen", "kitchen_to_food", "beauty_to_electronics", "electronics_to_beauty")
SEEDS = (42, 43, 44)
METRICS = tuple(f"{m}@{k}" for k in (5, 10, 20) for m in ("R", "N"))
MINIMUM_FREE_DISK_GIB = 24
DEFAULT_BATCH_ID = "amazon-expansion-dual-ndcg-disk24-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind(path, payload):
    if path.exists():
        if read(path) != payload:
            raise RuntimeError(f"Immutable experiment record changed: {path}")
    else:
        write_json_atomic(path, payload)
    return payload


def jobs():
    return [{"pair": pair, "seed": seed} for seed in SEEDS for pair in PAIRS]


def key(job):
    return f"{job['pair']}_seed{job['seed']}"


def folder_for(root, batch):
    return Path(root) / "artifacts/amazon_recommendation" / validate_batch_id(batch)


def job_folder(root, batch, job):
    return folder_for(root, batch) / "jobs" / job["pair"] / f"seed{job['seed']}"


def config_for(root, batch, job):
    base = shared.model_config(root, job["pair"], job["seed"])
    return replace(policy.AmazonConfig(**llm_training.config_payload(base)),
                   selection_metric=policy.MEAN_METRIC,
                   output_dir=job_folder(root, batch, job) / "training")


def source():
    identity = git_identity(ROOT)
    if identity["status_porcelain"]:
        raise RuntimeError("Recommendation tool checkout must be clean and committed")
    names = ("experiments/run_amazon_recommendation.py", "experiments/amazon_recommendation_policy.py",
             "experiments/prepare_amazon_assets.py", "experiments/run_gencdr_shared.py")
    return {"commit": identity["commit"], "files": {n: sha(ROOT / n) for n in names}}


def request(root, batch, alignment):
    return {"schema": "amazon-recommendation-request-v1", "root": str(root), "batch": batch,
            "jobs": jobs(), "alignment": str(alignment), "alignment_sha256": sha(alignment),
            "source": source(), "selection_policy": policy.SELECTION_POLICY,
            "checkpoint_metric": policy.MEAN_METRIC, "training_exact_tie": "earlier epoch",
            "maximum_workers": 2, "job_ram_bytes": 24 * 2**30,
            "minimum_free_disk_gib": MINIMUM_FREE_DISK_GIB,
            "minimum_free_mib": 12288, "maximum_gpu_utilization": 5,
            "test_released_only_after_all_validation": True, "paper_alignment_verified": False}


def asset_seals(root, alignment):
    missing = [p for p in PAIRS if not (shared.paths(root, p)["control"] / "ready.json").is_file()]
    if missing:
        print("WAIT_ASSETS " + ",".join(missing), flush=True)
        return None
    # The original seal writer is non-atomic. Wait for its controller lock
    # rather than interpreting a ready file observed mid-write as corruption.
    with ExitStack() as stack:
        try:
            stack.enter_context(recovery.operation_lock(
                root / "artifacts/diagnostics/amazon-expansion-assets-v1/controller"))
        except RuntimeError as error:
            if "holds this output" not in str(error):
                raise
            print("WAIT_ASSETS asset controller is finishing", flush=True)
            return None
        return verify_asset_seals(root, alignment)


def verify_asset_seals(root, alignment):
    report = assets.load_alignment(root, alignment)
    recipe = assets.reference_recipe(root)
    rules = read(shared.paths(root, "phones_to_electronics")["processed"] / "protocol.json")
    result = {}
    with assets.pair_registration():
        for pair in PAIRS:
            assets.verify_pair_data(root, pair, report, rules)
            assets.verify_pair_recipe(root, pair, recipe)
            shared.verify_ready(root, pair)
            p = shared.paths(root, pair)
            for file in (p["control"] / "ready.json", p["processed"] / "shared_manifest.json"):
                result[str(file.relative_to(root))] = sha(file)
    return result


def wait_for_assets(root, alignment, poll):
    while True:
        result = asset_seals(root, alignment)
        if result is not None:
            return result
        time.sleep(poll)


def active_workers(root, batch, processes=None):
    from experiments.extend_gencdr_run import owned_processes, option
    if processes is None:
        if sys.platform != "linux":
            return []  # Production GPU execution is on the Linux server.
        processes = owned_processes()
    result = []
    for process in processes:
        argv = process["argv"]
        if (process.get("state") == "Z"
                or not any(Path(a).name == "run_amazon_recommendation.py" for a in argv)
                or option(argv, "--worker") not in ("smoke", "validation", "test")
                or option(argv, "--batch-id") != batch):
            continue
        value = option(argv, "--root")
        if value and Path(value).resolve() == root.resolve():
            result.append(process)
    return result


def wait_for_workers(root, batch, poll):
    while True:
        workers = active_workers(root, batch)
        if not workers:
            return
        print("WAIT_EXISTING_WORKERS " + ",".join(str(p["pid"]) for p in workers), flush=True)
        time.sleep(poll)


def wait_for_disk(root, poll):
    """Resource-only gate: report actual capacity and wait without changing assets."""
    while True:
        free_bytes = shutil.disk_usage(root).free
        free_gib = free_bytes / 2**30
        print(f"DISK_CHECK free_gib={free_gib:.3f} required_gib={MINIMUM_FREE_DISK_GIB}", flush=True)
        if free_bytes >= MINIMUM_FREE_DISK_GIB * 2**30:
            return free_gib
        print(f"WAIT_DISK recheck={poll}s", flush=True)
        time.sleep(poll)


def resources(reserved):
    check_host_pressure()
    admit_host_budget(available=host_resources()["memory"].get("MemAvailable", 0),
                      reserved=reserved, required=24 * 2**30)


def run_queue(group, worker, status, *, max_workers=2, poll=60, inventory=query_gpu_inventory,
              resources=resources, sleep=time.sleep, completed=lambda job: False):
    pending = [dict(job) for job in group if not completed(job)]
    active, failure = {}, None
    status.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending or active:
            for future in [f for f in active if f.done()]:
                job, card = active.pop(future)
                record = {**job, "gpu_uuid": card["uuid"]}
                try:
                    future.result()
                    record["state"] = "complete"
                except ResourceUnavailable as error:
                    record.update(state="waiting", error=str(error))
                    pending.append(job)
                except Exception as error:
                    failure = str(error)
                    record.update(state="failed", error=failure)
                    print(f"WORKER_FAILED {key(job)} {failure}", flush=True)
                write_json_atomic(status / f"{key(job)}.json", record)
            if failure is not None:
                if active:
                    wait(active, return_when=FIRST_COMPLETED)
                    continue
                raise RuntimeError(f"Recommendation worker failed: {failure}")
            reason = "no idle GPU satisfying memory/UUID requirements"
            while pending and len(active) < max_workers:
                used = {c["uuid"] for _, c in active.values()}
                cards = sorted(inventory(), key=lambda c: (-c["memory_free_mib"], c["index"]))
                match = next(((j, c) for j in pending for c in cards
                    if c["uuid"] not in used and not c["pids"] and c["utilization"] <= 5
                    and c["memory_free_mib"] >= j.get("minimum_free_mib", 12288)
                    and (not j.get("gpu_uuid") or j["gpu_uuid"] == c["uuid"])), None)
                if match is None:
                    break
                try:
                    resources(len(active) * 24 * 2**30)
                except RuntimeError as error:
                    reason = str(error)
                    break
                if any(f.done() for f in active):
                    break
                job, card = match
                pending.remove(job)
                write_json_atomic(status / f"{key(job)}.json",
                                  {**job, "state": "running", "gpu_uuid": card["uuid"]})
                print(f"START {key(job)} GPU={card['index']} {card['uuid']}", flush=True)
                active[pool.submit(worker, job, card)] = (job, card)
            if active:
                wait(active, timeout=poll, return_when=FIRST_COMPLETED)
            elif pending:
                print(f"WAIT_GPU {reason}; pending={len(pending)}; recheck={poll}s", flush=True)
                sleep(poll)


def affinity_path(root, batch, job):
    return folder_for(root, batch) / "affinity" / f"{key(job)}.json"


def smoke_path(root, batch, pair):
    return folder_for(root, batch) / "smoke" / f"{pair}.json"


def completed_smoke(root, batch, pair):
    path = smoke_path(root, batch, pair)
    if not path.exists():
        return False
    saved = read(path)
    if (saved["ready_sha256"] != sha(shared.paths(root, pair)["control"] / "ready.json")
            or saved["source"] != source() or saved["report"]["optimizer_steps"] != 0):
        raise RuntimeError(f"Smoke identity changed: {pair}")
    training_memory_requirement(saved["report"])
    return True


def collect_binding(root, batch, job, *, allow_test=False):
    cfg = config_for(root, batch, job)
    folder = job_folder(root, batch, job)
    if not allow_test and any((cfg.output_dir / n).exists() for n in ("test.json", "test.lock")):
        raise RuntimeError("Test accessed before validation freeze")
    shared.verify_ready(root, job["pair"])
    runtime = folder / "runtime.json"
    if read(runtime) != llm_training.config_payload(cfg):
        raise RuntimeError("Runtime config changed")
    state, identity = validate_checkpoint(cfg, cfg.output_dir / "best_model.pt")
    manifest = read(cfg.output_dir / "manifest.json")
    if (manifest.get("complete") is not True or manifest["identity"] != state["identity"]
            or manifest["best_validation"] != state["validation"]
            or not 0 < state["epoch"] <= manifest["epochs_completed"]):
        raise RuntimeError("Incomplete or mismatched completed training")
    tuning = folder / "validation"
    search, verified = read(tuning / "search.json"), read(tuning / "verification.json")
    expected = recovery._identity(cfg, cfg.output_dir / "best_model.pt", identity, torch.device("cpu"))
    runtime_identity = search["identity"]["runtime"]
    uuid = read(affinity_path(root, batch, job))["gpu_uuid"]
    if (runtime_identity.get("uuid", "").lower().removeprefix("gpu-") != uuid.lower().removeprefix("gpu-")
            or runtime_identity.get("device") != "cuda:0"):
        raise RuntimeError("Validation GPU differs from admitted affinity")
    expected["runtime"] = runtime_identity
    hashes = {"snapshot_sha256": sha(tuning / "item_states.pt"),
              "cache_sha256": sha(tuning / "validation_scores.pt"), "search_sha256": sha(tuning / "search.json")}
    binding = {"identity": expected, **hashes}
    if (search["identity"] != expected or search.get("complete") is not True
            or search.get("split") != "validation" or search.get("selection_policy") != policy.SELECTION_POLICY
            or any(search[k] != hashes[k] for k in ("snapshot_sha256", "cache_sha256"))
            or verified.get("binding") != binding
            or not all(verified.get(k) is True for k in ("raw_scores_equal", "rankings_equal", "verified_all_metrics"))):
        raise RuntimeError("Validation selection/verification binding changed")
    if search["best"] != policy.select_fusion(search["results"]):
        raise RuntimeError("Validation selection differs from declared dual-NDCG rule")
    required = [runtime, cfg.output_dir / "best_model.pt", cfg.output_dir / "manifest.json",
                affinity_path(root, batch, job), smoke_path(root, batch, job["pair"]),
                shared.paths(root, job["pair"])["control"] / "ready.json"]
    required += [tuning / n for n in ("item_states.pt", "validation_scores.pt", "search.json", "verification.json")]
    return {**job, "files": {str(p.relative_to(root)): sha(p) for p in required},
            "test_identity": {**binding, "verification_sha256": sha(tuning / "verification.json"), "split": "test"}}


def freeze_bindings(folder, bindings):
    expected = {(j["pair"], j["seed"]) for j in jobs()}
    if len(bindings) != 12 or {(j["pair"], j["seed"]) for j in bindings} != expected:
        raise RuntimeError("Freeze requires all 12 direction/seed bindings")
    if any(not j.get("files") or not j.get("test_identity") for j in bindings):
        raise RuntimeError("Freeze requires full validation provenance")
    return bind(folder / "validation_freeze.json", {
        "schema": "amazon-12-job-validation-freeze-v1", "jobs": bindings,
        "declaration_sha256": sha(folder / "declaration.json"), "test_released": True})


def verify_freeze(root, batch):
    folder = folder_for(root, batch)
    if not (folder / "validation_freeze.json").exists():
        raise RuntimeError("All validation must be frozen before test")
    return freeze_bindings(folder, [collect_binding(root, batch, j, allow_test=True) for j in jobs()])


def record_test(folder, binding, report_path):
    report = read(report_path)
    if (report.get("state") != "complete" or report.get("split") != "test"
            or report.get("recovery_identity") != binding["test_identity"]):
        raise RuntimeError("Test result incomplete or identity changed; audit required")
    policy.metric_key(report["metrics"])
    return bind(folder / "test_completions" / f"{key(binding)}.json", {
        "pair": binding["pair"], "seed": binding["seed"], "report_sha256": sha(report_path),
        "freeze_sha256": sha(folder / "validation_freeze.json"), "test_identity": binding["test_identity"]})


def summarize_rows(rows):
    result = []
    for pair in PAIRS:
        subset = [r for r in rows if r["pair"] == pair]
        if len(subset) != 3 or {r["seed"] for r in subset} != set(SEEDS):
            raise RuntimeError(f"All three seeds are required: {pair}")
        metrics = {}
        for name in METRICS:
            values = [float(r["metrics"][name]) for r in subset]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise RuntimeError("Invalid test metric")
            metrics[name] = {"mean": statistics.mean(values), "std": statistics.stdev(values)}
        result.append({"pair": pair, "metrics": metrics})
    return result


def worker_action(args):
    root, batch = args.root.resolve(), args.batch_id
    job = {"pair": args.pair, "seed": args.seed}
    folder, run = folder_for(root, batch), job_folder(root, batch, job)
    declaration = read(folder / "declaration.json")
    if declaration["request"] != request(root, batch, args.alignment.resolve()):
        raise RuntimeError("Worker source/request changed")
    for relative, digest in declaration["assets"].items():
        if sha(root / relative) != digest:
            raise RuntimeError("Bound asset seal changed")
    shared.verify_gpu_binding()
    cfg = config_for(root, batch, job)
    if read(run / "runtime.json") != llm_training.config_payload(cfg):
        raise RuntimeError("Worker runtime differs from declaration")
    device = torch.device("cuda:0")
    with recovery.operation_lock(run / "operation"), assets.pair_registration(), policy.installed():
        shared.verify_ready(root, job["pair"])
        llm.check_cached_model(cfg)
        if args.worker == "smoke":
            report = llm.smoke(cfg, device)
            bind(smoke_path(root, batch, job["pair"]), {"report": report, "source": source(),
                "ready_sha256": sha(shared.paths(root, job["pair"])["control"] / "ready.json")})
        elif args.worker == "validation":
            if not completed_smoke(root, batch, job["pair"]):
                raise RuntimeError("Missing smoke gate")
            bind(affinity_path(root, batch, job), {**job, "gpu_uuid": os.environ["SIR_EXPECTED_GPU_UUID"]})
            if any((cfg.output_dir / n).exists() for n in ("test.json", "test.lock")):
                raise RuntimeError("Validation is closed after test access")
            llm_training.train(cfg, device)
            for stage in ("cache", "verify"):
                recovery.recover_stage(cfg, run / "validation", stage, device)
            collect_binding(root, batch, job)
        else:
            frozen = verify_freeze(root, batch)
            binding = next(j for j in frozen["jobs"] if (j["pair"], j["seed"]) == (job["pair"], job["seed"]))
            expected = read(affinity_path(root, batch, job))["gpu_uuid"]
            if expected != os.environ["SIR_EXPECTED_GPU_UUID"]:
                raise RuntimeError("Test GPU differs from frozen affinity")
            recovery.recover_stage(cfg, run / "validation", "test", device)
            record_test(folder, binding, cfg.output_dir / "test.json")


def run(args):
    root, batch, alignment = args.root.resolve(), args.batch_id, args.alignment.resolve()
    folder = folder_for(root, batch)
    with recovery.operation_lock(folder / "controller"):
        wait_for_workers(root, batch, args.poll_seconds)
        declaration_request = bind(folder / "request.json", request(root, batch, alignment))
        seals = wait_for_assets(root, alignment, args.poll_seconds)
        declaration = bind(folder / "declaration.json", {"request": declaration_request, "assets": seals,
            "configs": {key(j): llm_training.config_payload(config_for(root, batch, j)) for j in jobs()},
            "fusion_grid": parameter_settings(config_for(root, batch, jobs()[0]))})
        print("ALL_ASSETS_VERIFIED", flush=True)
        for job in jobs():
            cfg = config_for(root, batch, job)
            bind(job_folder(root, batch, job) / "runtime.json", llm_training.config_payload(cfg))
            llm.check_cached_model(cfg)
        wait_for_disk(root, args.poll_seconds)

        def launch(job, card, phase):
            cfg = config_for(root, batch, job)
            if read(job_folder(root, batch, job) / "runtime.json") != llm_training.config_payload(cfg):
                raise RuntimeError("Runtime config changed before admission")
            fresh = next((c for c in query_gpu_inventory() if c["uuid"] == card["uuid"]), None)
            if (fresh is None or fresh["pids"] or fresh["utilization"] > 5
                    or fresh["memory_free_mib"] < job.get("minimum_free_mib", 12288)):
                raise ResourceUnavailable("GPU changed occupancy before child launch")
            log = folder / "logs" / f"{key(job)}_{phase}.log"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase,
                "--root", str(root), "--alignment", str(alignment), "--batch-id", batch,
                "--pair", job["pair"], "--seed", str(job["seed"])]
            return run_commands([command], ROOT, log, card["uuid"])

        def queue(group, phase, completed):
            run_queue(group, lambda j, c: launch(j, c, phase), folder / "status" / phase,
                      max_workers=2, poll=args.poll_seconds, completed=completed)

        with assets.pair_registration(), policy.installed():
            queue([{"pair": p, "seed": 42} for p in PAIRS], "smoke",
                  lambda j: completed_smoke(root, batch, j["pair"]))
            work = []
            for job in jobs():
                job["minimum_free_mib"] = max(12288, training_memory_requirement(read(smoke_path(root, batch, job["pair"]))["report"]))
                affinity = affinity_path(root, batch, job)
                if affinity.exists():
                    job["gpu_uuid"] = read(affinity)["gpu_uuid"]
                work.append(job)

            def validation_done(job):
                path = job_folder(root, batch, job) / "validation/verification.json"
                if not path.exists():
                    return False
                collect_binding(root, batch, {"pair": job["pair"], "seed": job["seed"]},
                                allow_test=(folder / "validation_freeze.json").exists())
                return True

            queue(work, "validation", validation_done)
            frozen = freeze_bindings(folder, [collect_binding(root, batch, j,
                allow_test=(folder / "validation_freeze.json").exists()) for j in jobs()])
            print("VALIDATION_FROZEN 12/12; test released", flush=True)

            def test_done(job):
                cfg = config_for(root, batch, job)
                path = cfg.output_dir / "test.json"
                if not path.exists():
                    return False
                binding = next(j for j in frozen["jobs"] if (j["pair"], j["seed"]) == (job["pair"], job["seed"]))
                record_test(folder, binding, path)
                return True

            for job in work:
                job["gpu_uuid"] = read(affinity_path(root, batch, job))["gpu_uuid"]
            verify_freeze(root, batch)
            queue(work, "test", test_done)
            rows = []
            for job in jobs():
                test_done(job)
                report = read(config_for(root, batch, job).output_dir / "test.json")
                rows.append({**job, "metrics": report["metrics"]})
            bind(folder / "results.json", {"freeze_sha256": sha(folder / "validation_freeze.json"),
                "per_seed": rows, "directions": summarize_rows(rows), "paper_alignment_verified": False})
            print("AMAZON_RECOMMENDATION_COMPLETE " + str(folder / "results.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Data/lihb/SIR-CDR"))
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--action", choices=("run", "status"), default="run")
    parser.add_argument("--worker", choices=("smoke", "validation", "test"))
    parser.add_argument("--pair", choices=PAIRS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    validate_batch_id(args.batch_id)
    os.environ.update(HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.set_num_threads(1)
    if args.action == "status" and not args.worker:
        folder = folder_for(args.root.resolve(), args.batch_id)
        print(json.dumps({"request_exists": (folder / "request.json").exists(),
            "assets_bound": (folder / "declaration.json").exists(),
            "training_complete": sum((job_folder(args.root.resolve(), args.batch_id, j) / "training/manifest.json").exists() for j in jobs()),
            "validation_verified": sum((job_folder(args.root.resolve(), args.batch_id, j) / "validation/verification.json").exists() for j in jobs()),
            "frozen": (folder / "validation_freeze.json").exists(),
            "test_completions": len(list((folder / "test_completions").glob("*.json"))),
            "summary_exists": (folder / "results.json").exists()}, indent=2))
        return
    if args.alignment is None or args.poll_seconds < 5:
        parser.error("run requires --alignment and poll-seconds >=5")
    if args.worker:
        if not args.pair or args.seed is None or not os.environ.get("SIR_EXPECTED_GPU_UUID"):
            parser.error("worker requires pair, seed and admitted GPU UUID")
        worker_action(args)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        run(args)


if __name__ == "__main__":
    main()
