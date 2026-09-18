from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdr_framework.formal_training import _sha256, _write_json  # noqa: E402
from cdr_framework.text_config import TextCDRConfig  # noqa: E402
from cdr_framework.text_training import (  # noqa: E402
    VARIANTS,
    build_model,
    evaluate,
    load_rows,
    signature,
    variant_config,
    verify_checkpoint_identity,
)


def best_result(results: list[dict[str, float]]) -> dict[str, float]:
    """Select on validation NDCG@10 and prefer less generation on exact ties."""
    return max(results, key=lambda row: (row["NDCG@10"], -row["generation_weight"]))


@torch.no_grad()
def sweep(
    config: TextCDRConfig,
    device: torch.device,
    *,
    weights=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    batch_size: int = 16,
    rerank_candidates: int | None = None,
    fusion_norm: str = "none",
    allow_code_mismatch: bool = False,
) -> dict[str, object]:
    weights = sorted(set(float(weight) for weight in weights))
    if not weights or any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Weights must be finite and nonnegative.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    checkpoint_path = config.output_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training_identity = signature(config)
    ok, why = verify_checkpoint_identity(checkpoint, config, require_code=not allow_code_mismatch)
    if not ok:
        raise RuntimeError(f"Checkpoint {why}; use the unchanged training YAML.")

    runtime_base = replace(config, evaluation_batch_size=batch_size)
    if rerank_candidates is not None:
        runtime_base = replace(runtime_base, rerank_candidates=rerank_candidates)
    if fusion_norm != "none":
        # Inference-only dynamic attribute: NOT a dataclass field, so it stays out of
        # asdict()/signature and old checkpoints remain compatible. The config is a
        # frozen dataclass, so use object.__setattr__ for this runtime injection.
        object.__setattr__(runtime_base, "fusion_norm", fusion_norm)

    identity = {
        "training_identity": training_identity,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "script_sha256": _sha256(Path(__file__)),
        "split": "validation",
        "mode": "hybrid",
        "selection_metric": "NDCG@10",
        "rerank_candidates": runtime_base.rerank_candidates,
        "allow_code_mismatch": allow_code_mismatch,
        "fusion_norm": getattr(runtime_base, "fusion_norm", "none"),
        "retrieval_weight": runtime_base.retrieval_weight,
        "generation_weights": weights,
        "evaluation_batch_size": batch_size,
        "best_epoch": checkpoint["epoch"],
        "device": str(device),
        "torch_version": str(torch.__version__),
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    output = config.output_dir / f"validation_text_fusion_sweep_{run_id}.json"
    report: dict[str, object] = {"identity": identity, "results": [], "complete": False}
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("identity") != identity:
            raise RuntimeError("Fusion sweep identity mismatch.")

    model, catalog = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = load_rows(config, catalog, "validation")
    if not rows:
        raise ValueError("Validation split is empty.")

    results = report["results"]
    done = {row["generation_weight"] for row in results}
    for weight in weights:
        if weight in done:
            print(f"Reusing completed weight: {weight:g}", flush=True)
            continue
        runtime = replace(runtime_base, generation_weight=weight)
        model.config = runtime
        metrics = evaluate(model, rows, runtime, device, mode="hybrid")
        result = {
            "generation_weight": weight,
            "retrieval_weight": runtime.retrieval_weight,
            "rerank_candidates": runtime.rerank_candidates,
            **metrics,
        }
        results.append(result)
        report["best_so_far"] = best_result(results)
        _write_json(output, report)
        print(json.dumps(result, sort_keys=True), flush=True)

    report["complete"] = True
    report["best_validation_setting"] = best_result(results)
    _write_json(output, report)
    print("Best validation setting: " + json.dumps(report["best_validation_setting"], sort_keys=True))
    print(f"Report: {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only Text-CDR retrieval/generation fusion sweep; no retraining."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", nargs="+", type=float, default=[0, 0.1, 0.25, 0.5, 1, 2])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-root", type=Path,
                        help="Override config.output_dir before variant suffix (same as run_text_cdr.py).")
    parser.add_argument("--rerank-candidates", type=int, default=None,
                        help="Override rerank_candidates at evaluation time (training signature unchanged).")
    parser.add_argument("--fusion-norm", choices=("none", "softmax"), default="none",
                        help="Normalize retrieval/generation scores before fusion (runtime only, no retrain).")
    parser.add_argument("--allow-code-mismatch", action="store_true",
                        help="Inference-only: accept newer code while config/artifacts are still verified; recorded in report.")
    args = parser.parse_args()

    config = TextCDRConfig.from_yaml(args.config)
    config = replace(
        config,
        **{
            name: path if path.is_absolute() else ROOT / path
            for name in ("processed_dir", "tokenizer_dir", "output_dir")
            for path in [getattr(config, name)]
        },
    )
    if args.output_root is not None:
        config = replace(config, output_dir=args.output_root.resolve())
    config = variant_config(config, args.variant, args.seed)
    sweep(config, torch.device(args.device), weights=args.weights, batch_size=args.batch_size,
          rerank_candidates=args.rerank_candidates, fusion_norm=args.fusion_norm,
          allow_code_mismatch=args.allow_code_mismatch)


if __name__ == "__main__":
    main()
