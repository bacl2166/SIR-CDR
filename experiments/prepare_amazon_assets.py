"""Generate new Amazon embeddings and SIDs with the completed Phones recipe."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdr_framework.datasets.gencdr_protocol import sha, verify_shared
from cdr_framework.experiment_config import TokenizerTrainingConfig
from cdr_framework.experiment_control import query_gpu_inventory, write_json_atomic
from experiments import run_gencdr_shared as shared
from experiments.audit_cross_domain_candidates import PAIRS
from experiments.recover_llm_evaluation import operation_lock
from experiments.run_gencdr_six import check_host_pressure, run_commands, schedule_jobs

NEW_PAIRS = dict(PAIRS)
NEW_PAIRS.update({"_to_".join(reversed(name.split("_to_"))): tuple(reversed(domains))
                  for name, domains in PAIRS.items()})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def pair_registration():
    previous = dict(shared.PAIRS)
    try:
        shared.PAIRS.update(NEW_PAIRS)
        yield
    finally:
        shared.PAIRS.clear()
        shared.PAIRS.update(previous)


def reference_recipe(root):
    shared.verify_ready(root, "phones_to_electronics")
    control = shared.paths(root, "phones_to_electronics")["control"]
    embedding = read(control / "embedding_model.json")
    if not re.fullmatch(r"[0-9a-f]{40}", embedding["revision"]):
        raise RuntimeError("Reference embedding model must be pinned")
    expected = asdict(TokenizerTrainingConfig())
    actual = read(control / "tokenizer_execution.json")["config"]
    paths = {"embeddings_path", "item_texts_path", "output_dir"}
    if {k: v for k, v in actual.items() if k not in paths} != {
            k: v for k, v in expected.items() if k not in paths}:
        raise RuntimeError("Phones tokenizer recipe differs from the shared tokenizer implementation")
    return {"embedding": embedding,
            "tokenizer": {k: v for k, v in actual.items() if k not in paths},
            "embedding_receipt_sha256": sha(control / "embedding_model.json"),
            "tokenizer_receipt_sha256": sha(control / "tokenizer_execution.json")}


def bind_embedding(root, pair, recipe):
    control = shared.paths(root, pair)["control"]
    destination = control / "embedding_model.json"
    payload = {**recipe["embedding"], "identity": shared.asset_identity(root, pair)}
    if destination.exists():
        if read(destination) != payload:
            raise RuntimeError("Embedding receipt differs from the bound Phones model")
    else:
        write_json_atomic(destination, payload)


def bind(path, payload):
    if path.exists():
        if read(path) != payload:
            raise RuntimeError(f"Asset execution binding changed: {path}")
    else:
        write_json_atomic(path, payload)


def verify_pair_data(root, pair, alignment, reference_rules):
    processed = shared.paths(root, pair)["processed"]
    verify_shared(processed)
    if sha(processed / "shared_manifest.json") != alignment["directions"][pair]["shared_manifest_sha256"]:
        raise RuntimeError(f"Processed data differs from alignment report: {pair}")
    current = read(processed / "protocol.json")
    identity = current["identity"]
    if (identity["source_domain"], identity["target_domain"]) != NEW_PAIRS[pair]:
        raise RuntimeError(f"Processed domain roles differ: {pair}")
    if {k: v for k, v in current.items() if k != "identity"} != {
            k: v for k, v in reference_rules.items() if k != "identity"}:
        raise RuntimeError(f"Data protocol differs from Phones: {pair}")


def load_alignment(root, path):
    report = read(path)
    reference = shared.paths(root, "phones_to_electronics")["processed"]
    if (report.get("complete") is not True or report.get("schema") != "amazon-expansion-phones-alignment-v1"
            or set(report.get("directions", {})) != set(NEW_PAIRS)
            or report.get("reference_sha256") != sha(reference / "shared_manifest.json")):
        raise RuntimeError("A complete matching ALIGNMENT_OK report is required")
    return report


def verify_pair_recipe(root, pair, recipe):
    bind_embedding(root, pair, recipe)
    execution = read(shared.paths(root, pair)["control"] / "tokenizer_execution.json")
    paths = {"embeddings_path", "item_texts_path", "output_dir"}
    if {k: v for k, v in execution["config"].items() if k not in paths} != recipe["tokenizer"]:
        raise RuntimeError(f"Completed tokenizer recipe differs from Phones: {pair}")


def select_cards(cards, pending):
    required = {job["gpu_uuid"] for job in pending if job.get("gpu_uuid")}
    available = [c for c in cards if not c["pids"] and c["utilization"] <= 5
                 and c["memory_free_mib"] >= 24576]
    available.sort(key=lambda c: (c["uuid"] not in required, -c["memory_free_mib"], c["index"]))
    return available[:6]


def run_stages(root, pair, gpu, alignment_path):
    control = shared.paths(root, pair)["control"]
    log = root / "logs/amazon-expansion-assets-v1" / f"{pair}.log"
    for action in ("embedding", "tokenize"):
        card = next(c for c in query_gpu_inventory() if c["index"] == gpu)
        if card["pids"] or card["utilization"] > 5 or card["memory_free_mib"] < 24576:
            raise RuntimeError(f"GPU {gpu} is no longer available before {action}")
        bind(control / "asset_gpu.json", {"gpu_uuid": card["uuid"]})
        command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                   "--alignment", str(alignment_path), "--stage", action, "--pair", pair]
        result = run_commands([command], ROOT, log, card["uuid"])
    return result


def stage(root, pair, action, alignment_path):
    shared.verify_gpu_binding()
    recipe = reference_recipe(root)
    alignment = load_alignment(root, alignment_path)
    with pair_registration():
        control = shared.paths(root, pair)["control"]
        with operation_lock(control / "controller"):
            rules = read(shared.paths(root, "phones_to_electronics")["processed"] / "protocol.json")
            verify_pair_data(root, pair, alignment, rules)
            bind_embedding(root, pair, recipe)
            if action == "embedding":
                shared.embedding(root, pair, "cuda:0")
            else:
                shared.tokenize(root, pair, "cuda:0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Data/lihb/SIR-CDR"))
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--stage", choices=("embedding", "tokenize"))
    parser.add_argument("--pair", choices=tuple(NEW_PAIRS))
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.stage:
        if not args.pair or not os.environ.get("SIR_EXPECTED_GPU_UUID"):
            parser.error("Internal stage requires a pair and admitted GPU environment")
        stage(root, args.pair, args.stage, args.alignment.resolve())
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    directory = root / "artifacts/diagnostics/amazon-expansion-assets-v1"
    with operation_lock(directory / "controller"), pair_registration():
        recipe = reference_recipe(root)
        alignment = load_alignment(root, args.alignment)
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download("Qwen/Qwen3-Embedding-8B",
                        revision=recipe["embedding"]["revision"], local_files_only=True))
        if sha(snapshot / "config.json") != recipe["embedding"]["model_config_sha256"]:
            raise RuntimeError("Cached embedding model differs from the Phones model")
        rules = read(shared.paths(root, "phones_to_electronics")["processed"] / "protocol.json")
        pending = []
        for pair in NEW_PAIRS:
            paths = shared.paths(root, pair)
            verify_pair_data(root, pair, alignment, rules)
            bind(paths["control"] / "expansion_recipe.json", {
                "pair": pair, "domains": list(NEW_PAIRS[pair]), "tool_sha256": sha(__file__),
                "reference": recipe, "protocol_sha256": sha(paths["processed"] / "shared_manifest.json"),
                "alignment_sha256": sha(args.alignment),
            })
            if (paths["control"] / "ready.json").is_file():
                verify_pair_recipe(root, pair, recipe)
                shared.verify_ready(root, pair)
                print(f"REUSED {pair}", flush=True)
                continue
            job = {"pair": pair, "seed": 42, "minimum_free_mib": 24576,
                   "host_ram_bytes": 24 * 2**30}
            affinity = paths["control"] / "asset_gpu.json"
            if affinity.exists():
                job["gpu_uuid"] = read(affinity)["gpu_uuid"]
            pending.append(job)
        if not pending:
            print("AMAZON_ASSETS_COMPLETE", flush=True)
            return
        available = select_cards(query_gpu_inventory(), pending)
        if not available:
            print("WAIT_GPU No idle GPU with at least 24 GiB free", flush=True)
            return

        def worker(job, gpu):
            pair = job["pair"]
            result = run_stages(root, pair, gpu, args.alignment.resolve())
            shared.seal(root, pair)
            shared.verify_ready(root, pair)
            verify_pair_recipe(root, pair, recipe)
            print(f"ASSETS_READY {pair}", flush=True)
            return result

        results = schedule_jobs(pending, [c["index"] for c in available],
                                min(args.max_workers, len(available)), directory / "status", worker,
                                resource_check=check_host_pressure)
        if any(result["returncode"] for result in results):
            raise RuntimeError("Asset worker failed; inspect its log and rerun to resume")
        print("AMAZON_ASSETS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
