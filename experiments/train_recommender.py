from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cdr_framework.experiment_config import RecommendationTrainingConfig  # noqa: E402
from cdr_framework.formal_training import run_formal_training  # noqa: E402


def _repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SIR-CDR on the formal domain pair.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = RecommendationTrainingConfig.from_yaml(args.config)
        config = replace(
            config,
            processed_dir=_repository_path(config.processed_dir),
            tokenizer_dir=_repository_path(config.tokenizer_dir),
            output_dir=_repository_path(config.output_dir),
        )

        def report(record: dict) -> None:
            line = (
                f"Epoch {record['epoch']:03d}/{config.max_epochs}: "
                f"total={record['total']:.6f} "
                f"gen={record['generation']:.6f} "
                f"retrieval={record['retrieval']:.6f} "
                f"cpf={record['cpf']:.6f}"
            )
            if "validation" in record:
                metrics = record["validation"]
                line += " " + " ".join(
                    f"val_{name}={metrics[name]:.6f}"
                    for name in ("HR@10", "NDCG@10", "MRR@10")
                    if name in metrics
                )
            print(line, flush=True)

        result = run_formal_training(
            config, device=args.device, force=args.force, report=report
        )
        if result.already_complete:
            print("Recommendation output is already complete; training was not repeated.")
        print(f"Epochs completed: {result.epochs_completed}")
        print(f"Best validation: {result.best_validation}")
        print(f"Output: {result.output_dir}")
        return 0
    except Exception as error:
        print(f"Recommendation training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
