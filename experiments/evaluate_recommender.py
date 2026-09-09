from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cdr_framework.experiment_config import RecommendationTrainingConfig  # noqa: E402
from cdr_framework.formal_training import evaluate_saved_model  # noqa: E402


def _repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SIR-CDR over the full target catalog.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args(argv)
    try:
        config = RecommendationTrainingConfig.from_yaml(args.config)
        config = replace(
            config,
            processed_dir=_repository_path(config.processed_dir),
            tokenizer_dir=_repository_path(config.tokenizer_dir),
            output_dir=_repository_path(config.output_dir),
        )
        metrics = evaluate_saved_model(config, split=args.split, device=args.device)
        output_path = config.output_dir / f"{args.split}_metrics.json"
        output_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"Metrics: {output_path}")
        return 0
    except Exception as error:
        print(f"Recommendation evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
