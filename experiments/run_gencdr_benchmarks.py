from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdr_framework.formal_training import load_fixed_catalog  # noqa: E402
from cdr_framework.text_config import TextCDRConfig  # noqa: E402
from cdr_framework.text_training import (  # noqa: E402
    evaluation_identity,
    evaluate_checkpoint,
    load_rows,
    train,
    variant_config,
)
from experiments.sweep_text_fusion import sweep  # noqa: E402


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    config_path: Path
    processed_dir: Path
    tokenizer_dir: Path
    target_domain: str


BENCHMARKS = {
    "sports_to_clothing": BenchmarkSpec(
        "sports_to_clothing",
        Path("configs/text_sports_clothing_v4_p0.yaml"),
        Path("data/processed/sports_to_clothing"),
        Path("artifacts/tokenizer/qwen3_8b/sports_to_clothing"),
        "Clothing_Shoes_and_Jewelry",
    ),
    "phones_to_electronics": BenchmarkSpec(
        "phones_to_electronics",
        Path("configs/text_phones_electronics_v4_p0.yaml"),
        Path("data/processed/phones_to_electronics"),
        Path("artifacts/tokenizer/qwen3_8b/phones_to_electronics"),
        "Electronics",
    ),
    "books_to_movies": BenchmarkSpec(
        "books_to_movies",
        Path("configs/text_books_movies_v4_p0.yaml"),
        Path("data/processed/books_to_movies"),
        Path("artifacts/tokenizer/qwen3_8b/books_to_movies"),
        "Movies",
    ),
}

PROCESSED_FILES = ("mappings.json", "train.jsonl", "validation.jsonl", "test.jsonl")
TOKENIZER_FILES = ("item_latents.pt", "semantic_ids.pt", "tokenizer.pt", "quality_report.json")
METRIC_COLUMNS = (
    "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20",
    "MRR@5", "MRR@10", "MRR@20", "coverage", "examples", "seconds",
)


class AssetPreflightError(RuntimeError):
    pass


def validate_repository_identity(head: str, upstream: str, tracked_changes: str) -> dict[str, str]:
    if head != upstream:
        raise AssetPreflightError(
            f"Repository revision mismatch: HEAD={head}, upstream={upstream}; run git pull --ff-only"
        )
    if tracked_changes.strip():
        raise AssetPreflightError(
            "Tracked files are modified; stash or commit them before starting benchmarks:\n"
            + tracked_changes.strip()
        )
    return {"commit": head, "upstream": upstream, "tracked_changes": "clean"}


def repository_identity(root: Path) -> dict[str, str]:
    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ("git", *arguments), cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()

    return validate_repository_identity(
        git("rev-parse", "HEAD"),
        git("rev-parse", "@{u}"),
        git("status", "--porcelain", "--untracked-files=no"),
    )


def _absolute_config(path: Path, seed: int) -> TextCDRConfig:
    config = TextCDRConfig.from_yaml(path)
    config = replace(
        config,
        **{
            name: value if value.is_absolute() else ROOT / value
            for name in ("processed_dir", "tokenizer_dir", "output_dir")
            for value in [getattr(config, name)]
        },
    )
    return variant_config(config, "full", seed)


def preflight_assets(root: Path, specs: tuple[BenchmarkSpec, ...]) -> dict[str, dict[str, object]]:
    missing: dict[str, list[str]] = {}
    for spec in specs:
        required = [root / spec.config_path]
        required.extend(root / spec.processed_dir / name for name in PROCESSED_FILES)
        required.extend(root / spec.tokenizer_dir / name for name in TOKENIZER_FILES)
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            missing[spec.name] = absent
    if missing:
        lines = ["Three-dataset preflight failed; no training was started."]
        for name, paths in missing.items():
            lines.append(f"{name}:")
            lines.extend(f"  missing: {path}" for path in paths)
        raise AssetPreflightError("\n".join(lines))

    report: dict[str, dict[str, object]] = {}
    for spec in specs:
        config_path = root / spec.config_path
        config = TextCDRConfig.from_yaml(config_path)
        config = replace(
            config,
            processed_dir=root / config.processed_dir,
            tokenizer_dir=root / config.tokenizer_dir,
            output_dir=root / config.output_dir,
        )
        quality = json.loads((config.tokenizer_dir / "quality_report.json").read_text(encoding="utf-8"))
        if quality.get("passed") is not True:
            raise AssetPreflightError(f"{spec.name}: tokenizer quality report did not pass")
        tokenizer = torch.load(config.tokenizer_dir / "tokenizer.pt", map_location="cpu", weights_only=True)
        if tokenizer.get("schema_version") != 2:
            raise AssetPreflightError(f"{spec.name}: tokenizer schema_version must be 2")
        mappings = json.loads((config.processed_dir / "mappings.json").read_text(encoding="utf-8"))
        if mappings.get("target_domain") != spec.target_domain:
            raise AssetPreflightError(
                f"{spec.name}: target_domain={mappings.get('target_domain')!r}, expected {spec.target_domain!r}"
            )
        catalog = load_fixed_catalog(config)
        counts = {split: len(load_rows(config, catalog, split)) for split in ("train", "validation", "test")}
        report[spec.name] = {
            "target_domain": spec.target_domain,
            "rows": counts,
            "items": len(catalog.item_latents),
            "target_items": len(catalog.target_item_ids),
            "latent_shape": list(catalog.item_latents.shape),
            "semantic_id_shape": list(catalog.semantic_ids.shape),
            "codebook_shape": list(catalog.codebooks.shape),
        }
    return report


def select_validation_setting(reports: list[dict[str, object]]) -> dict[str, object]:
    if not reports:
        raise ValueError("No completed validation sweep reports")
    report = max(
        reports,
        key=lambda row: (
            row["best_validation_setting"]["NDCG@10"],
            -row["best_validation_setting"]["generation_score_weight"],
        ),
    )
    best = dict(report["best_validation_setting"])
    best["fusion_normalization"] = report["identity"]["fusion_normalization"]
    return best


def collect_metrics(
    results: dict[str, dict[str, object]], output_dir: Path, *, seed: int
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"gencdr_three_datasets_seed{seed}.json"
    csv_path = output_dir / f"gencdr_three_datasets_seed{seed}.csv"
    json_path.write_text(
        json.dumps({"seed": seed, "datasets": results}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("dataset", "epoch", "checkpoint_sha256", *METRIC_COLUMNS),
        )
        writer.writeheader()
        for name, result in results.items():
            metrics = result["metrics"]
            writer.writerow({
                "dataset": name,
                "epoch": result["epoch"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                **{metric: metrics.get(metric, "") for metric in METRIC_COLUMNS},
            })
    return json_path, csv_path


def _evaluate_once(config: TextCDRConfig, device: torch.device) -> dict[str, object]:
    result_path = config.output_dir / "test_hybrid.json"
    current_identity = evaluation_identity(
        config, config.output_dir / "best_model.pt", "test", "hybrid"
    )
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("evaluation_identity") != current_identity:
            raise RuntimeError(
                f"Existing test result has a different evaluation identity: {result_path}"
            )
        print(f"Reusing immutable test result: {result_path}", flush=True)
        return result
    return evaluate_checkpoint(config, device, split="test", mode="hybrid")


def evaluate_pair(
    config: TextCDRConfig,
    device: torch.device,
    *,
    weights: tuple[float, ...],
    normalizations: tuple[str, ...],
    batch_size: int,
) -> dict[str, object]:
    reports = [
        sweep(
            config,
            device,
            weights=weights,
            batch_size=batch_size,
            fusion_normalization=normalization,
        )
        for normalization in normalizations
    ]
    selected = select_validation_setting(reports)
    runtime = replace(
        config,
        retrieval_score_weight=float(selected["retrieval_score_weight"]),
        generation_score_weight=float(selected["generation_score_weight"]),
        fusion_normalization=str(selected["fusion_normalization"]),
    )
    selection_path = config.output_dir / "selected_validation_setting.json"
    selection_path.write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    return _evaluate_once(runtime, device)


def _selected_specs(names: list[str] | None) -> tuple[BenchmarkSpec, ...]:
    return tuple(BENCHMARKS[name] for name in (names or list(BENCHMARKS)))


def action_requires_device(action: str) -> bool:
    return action in {"train", "evaluate", "all"}


def run(args: argparse.Namespace) -> dict[str, object]:
    specs = _selected_specs(args.pairs)
    repository = repository_identity(ROOT)
    report = preflight_assets(ROOT, specs)
    print(json.dumps({"repository": repository, "preflight": report}, indent=2, sort_keys=True), flush=True)
    if args.action == "preflight":
        return {"repository": repository, "preflight": report}
    if action_requires_device(args.action) and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device) if action_requires_device(args.action) else torch.device("cpu")
    configs = {spec.name: _absolute_config(ROOT / spec.config_path, args.seed) for spec in specs}
    if args.action in {"train", "all"}:
        for name, config in configs.items():
            print(f"=== TRAIN {name} ===", flush=True)
            train(config, device)
        if args.action == "train":
            return {"repository": repository, "preflight": report}
    results: dict[str, dict[str, object]] = {}
    if args.action in {"evaluate", "all"}:
        for name, config in configs.items():
            print(f"=== VALIDATE AND TEST {name} ===", flush=True)
            results[name] = evaluate_pair(
                config,
                device,
                weights=tuple(args.weights),
                normalizations=tuple(args.normalizations),
                batch_size=args.evaluation_batch_size,
            )
    else:
        for name, config in configs.items():
            path = config.output_dir / "test_hybrid.json"
            if not path.is_file():
                raise RuntimeError(f"Missing completed test result: {path}")
            results[name] = json.loads(path.read_text(encoding="utf-8"))
    json_path, csv_path = collect_metrics(
        results, ROOT / "artifacts/text_cdr_v4/benchmark_reports", seed=args.seed
    )
    print(f"Summary JSON: {json_path}", flush=True)
    print(f"Summary CSV: {csv_path}", flush=True)
    return {"repository": repository, "preflight": report, "results": results,
            "json": str(json_path), "csv": str(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the three GenCDR paper dataset pairs through the SIR-CDR v4 protocol."
    )
    parser.add_argument("--action", choices=("preflight", "train", "evaluate", "all", "summarize"), default="preflight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", nargs="+", choices=tuple(BENCHMARKS))
    parser.add_argument("--weights", nargs="+", type=float, default=[0, 0.1, 0.25, 0.5, 1, 2])
    parser.add_argument("--normalizations", nargs="+", choices=("none", "log_softmax", "zscore"),
                        default=["none", "log_softmax", "zscore"])
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.evaluation_batch_size < 1:
        parser.error("--evaluation-batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
