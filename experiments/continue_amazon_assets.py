"""Continue assets by stage, refreshing idle GPUs and preserving completed embeddings."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from cdr_framework.datasets.gencdr_protocol import sha
from cdr_framework.embeddings.qwen import load_item_texts
from cdr_framework.experiment_control import (
    host_resources, query_gpu_inventory, require_embedding_recipe, write_json_atomic,
)
from experiments import prepare_amazon_assets as original
from experiments import run_gencdr_shared as shared
from experiments.recover_llm_evaluation import operation_lock
from experiments.run_gencdr_six import admit_host_budget, check_host_pressure, run_commands


class ResourceUnavailable(RuntimeError):
    """Prelaunch resource race; no child has been started."""


def embedding_complete(root, pair):
    paths = shared.paths(root, pair)
    manifest_path = paths["embedding"] / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = original.read(manifest_path)
    model = original.read(paths["control"] / "embedding_model.json")
    texts = paths["processed"] / "item_texts.jsonl"
    items = load_item_texts(texts)
    ids = [item.item_id for item in items]
    count = len(items)
    require_embedding_recipe(manifest, {
        "input_sha256": sha(texts), "model_revision": model["revision"],
        "model_config_sha256": model["model_config_sha256"], "dimension": 768,
        "normalized": True, "max_sequence_length": 512, "item_count": count,
    })
    if any(manifest.get(key) != value for key, value in shared.asset_identity(root, pair).items()):
        raise RuntimeError(f"Completed embedding provenance differs: {pair}")
    if any(manifest.get(key) != value for key, value in {
            "schema_version": 1, "model": "Qwen/Qwen3-Embedding-8B",
            "provider": "qwen3_local", "batch_size": 2,
            "dtype": "float32", "padding_row_is_zero": True}.items()):
        raise RuntimeError(f"Completed embedding recipe differs: {pair}")
    if ids != list(range(1, count + 1)) or original.read(paths["embedding"] / "item_index.json") != {
            "padding_item_id": 0, "row_is_item_id": True, "item_ids": ids}:
        raise RuntimeError(f"Completed embedding item index differs: {pair}")
    matrix_path = paths["embedding"] / "text_embeddings.pt"
    digest = sha(matrix_path)
    matrix = torch.load(matrix_path, map_location="cpu", weights_only=True)
    if (matrix.dtype != torch.float32 or list(matrix.shape) != [count + 1, 768]
            or not torch.isfinite(matrix).all() or torch.count_nonzero(matrix[0]).item()):
        raise RuntimeError(f"Invalid completed embedding matrix: {pair}")
    if not torch.allclose(matrix[1:].norm(dim=1), torch.ones(count), atol=1e-4, rtol=1e-4):
        raise RuntimeError(f"Completed embedding vectors are not normalized: {pair}")
    for batch, start in enumerate(range(0, count, 2)):
        chunk = torch.load(paths["embedding"] / "chunks" / f"batch-{batch:06d}.pt",
                           map_location="cpu", weights_only=True)
        expected_ids = ids[start:start + 2]
        if (chunk["item_ids"].tolist() != expected_ids
                or not torch.equal(chunk["embeddings"].to(torch.float32), matrix[expected_ids])):
            raise RuntimeError(f"Cached embedding batch differs from completed matrix: {pair}/{batch}")
        if (batch + 1) % 1000 == 0:
            print(f"VERIFY_EMBEDDING {pair} chunks={batch + 1}/{manifest['completed_batches']}", flush=True)
    if sha(matrix_path) != digest:
        raise RuntimeError(f"Completed embedding changed during verification: {pair}")
    return digest


def resources(reserved):
    check_host_pressure()
    admit_host_budget(available=host_resources()["memory"].get("MemAvailable", 0),
                      reserved=reserved, required=24 * 2**30)


def schedule_stages(jobs, worker, status, *, max_workers=2, poll=60,
                    inventory=query_gpu_inventory, resources=resources, sleep=time.sleep):
    pending = [dict(job) for job in jobs]
    active = {}
    failure = None
    status.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending or active:
            finished = [future for future in active if future.done()]
            for future in finished:
                job, card = active.pop(future)
                record = {**job, "gpu": card["index"], "gpu_uuid": card["uuid"]}
                try:
                    digest = future.result()
                    record["state"] = "complete"
                    if job["phase"] == "embedding":
                        pending.insert(0, {"pair": job["pair"], "phase": "tokenize",
                                           "embedding_sha256": digest})
                except ResourceUnavailable as error:
                    record.update(state="waiting", error=str(error))
                    pending.append(job)
                    print(f"WAIT_GPU {job['pair']} {job['phase']} {error}", flush=True)
                except Exception as error:
                    record.update(state="failed", error=str(error))
                    failure = str(error)
                    print(f"WORKER_FAILED {job['pair']} {job['phase']} {error}", flush=True)
                write_json_atomic(status / f"{job['pair']}_{job['phase']}.json", record)
            if failure is not None:
                if active:
                    wait(active, return_when=FIRST_COMPLETED)
                    continue
                raise RuntimeError(f"Asset stage failed: {failure}")
            reason = "No idle GPU with at least 24 GiB free"
            while pending and len(active) < max_workers:
                used = {card["uuid"] for _, card in active.values()}
                cards = [card for card in inventory() if card["uuid"] not in used
                         and not card["pids"] and card["utilization"] <= 5
                         and card["memory_free_mib"] >= 24576]
                cards.sort(key=lambda card: (-card["memory_free_mib"], card["index"]))
                if not cards:
                    break
                try:
                    resources(len(active) * 24 * 2**30)
                except RuntimeError as error:
                    reason = str(error)
                    break
                # A sibling may have failed while the external resource probes ran.
                if any(future.done() for future in active):
                    break
                job, card = pending.pop(0), cards[0]
                write_json_atomic(status / f"{job['pair']}_{job['phase']}.json",
                                  {**job, "gpu": card["index"], "gpu_uuid": card["uuid"], "state": "running"})
                print(f"START {job['pair']} {job['phase']} GPU={card['index']} {card['uuid']}", flush=True)
                active[pool.submit(worker, job, card)] = (job, card)
            if active:
                wait(active, timeout=poll, return_when=FIRST_COMPLETED)
            elif pending:
                print(f"WAIT_GPU {reason}; recheck in {poll}s", flush=True)
                sleep(poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Data/lihb/SIR-CDR"))
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("poll-seconds must be at least 5")
    root, alignment_path = args.root.resolve(), args.alignment.resolve()
    os.environ.update(HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    torch.set_num_threads(1)
    directory = root / "artifacts/diagnostics/amazon-expansion-assets-v1"
    with operation_lock(directory / "controller"), original.pair_registration():
        recipe = original.reference_recipe(root)
        alignment = original.load_alignment(root, alignment_path)
        rules = original.read(shared.paths(root, "phones_to_electronics")["processed"] / "protocol.json")
        jobs = []
        for pair in original.NEW_PAIRS:
            paths = shared.paths(root, pair)
            original.verify_pair_data(root, pair, alignment, rules)
            # Preserve the exact existing recipe receipt and its original tool hash.
            original.bind(paths["control"] / "expansion_recipe.json", {
                "pair": pair, "domains": list(original.NEW_PAIRS[pair]),
                "tool_sha256": sha(original.__file__), "reference": recipe,
                "protocol_sha256": sha(paths["processed"] / "shared_manifest.json"),
                "alignment_sha256": sha(alignment_path),
            })
            original.bind_embedding(root, pair, recipe)
            if (paths["control"] / "ready.json").is_file():
                original.verify_pair_recipe(root, pair, recipe)
                shared.verify_ready(root, pair)
                print(f"REUSED {pair}", flush=True)
            else:
                digest = embedding_complete(root, pair)
                phase = "tokenize" if digest else "embedding"
                jobs.append({"pair": pair, "phase": phase, "embedding_sha256": digest or None})
                print(f"PENDING {pair} {phase}", flush=True)

        def worker(job, card):
            pair, phase = job["pair"], job["phase"]
            control = shared.paths(root, pair)["control"]
            matrix_path = shared.paths(root, pair)["embedding"] / "text_embeddings.pt"
            if phase == "tokenize" and sha(matrix_path) != job["embedding_sha256"]:
                raise RuntimeError(f"Verified embedding changed while queued: {pair}")
            fresh = next((c for c in query_gpu_inventory() if c["uuid"] == card["uuid"]), None)
            if (fresh is None or fresh["pids"] or fresh["utilization"] > 5
                    or fresh["memory_free_mib"] < 24576):
                raise ResourceUnavailable(f"GPU {card['index']} changed occupancy before launch")
            legacy = control / "asset_gpu.json"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            receipt = control / "stage_executions" / f"{phase}-{stamp}.json"
            record = {**job, "gpu": fresh, "controller_sha256": sha(__file__),
                      "legacy_affinity_sha256": sha(legacy) if legacy.exists() else None,
                      "embedding_reused": phase == "tokenize", "state": "running"}
            write_json_atomic(receipt, record)
            log = root / "logs/amazon-expansion-assets-v1" / f"{pair}.log"
            command = [sys.executable, str(Path(original.__file__).resolve()), "--root", str(root),
                       "--alignment", str(alignment_path), "--stage", phase, "--pair", pair]
            try:
                run_commands([command], ROOT, log, card["uuid"])
                if phase == "embedding":
                    digest = embedding_complete(root, pair)
                    if not digest:
                        raise RuntimeError("Embedding stage produced no completed manifest")
                    record["embedding_sha256"] = digest
                else:
                    execution = original.read(control / "tokenizer_execution.json")
                    if (execution["embedding_sha256"] != job["embedding_sha256"]
                            or sha(matrix_path) != job["embedding_sha256"]):
                        raise RuntimeError(f"Tokenizer used a different embedding than verified: {pair}")
                    original.verify_pair_recipe(root, pair, recipe)
                    shared.seal(root, pair)
                    shared.verify_ready(root, pair)
                    print(f"ASSETS_READY {pair}", flush=True)
                record["state"] = "complete"
            except Exception as error:
                record.update(state="failed", error=str(error))
                raise
            finally:
                write_json_atomic(receipt, record)
            return digest if phase == "embedding" else None

        schedule_stages(jobs, worker, directory / "continuation_status",
                        max_workers=args.max_workers, poll=args.poll_seconds)
        print("AMAZON_ASSETS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
