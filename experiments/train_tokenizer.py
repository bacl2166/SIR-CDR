from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cdr_framework.experiment_config import TokenizerTrainingConfig  # noqa: E402
from cdr_framework.tokenization.training import (  # noqa: E402
    load_tokenizer_dataset,
    train_semantic_tokenizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the domain-adaptive Semantic ID tokenizer."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device such as cuda or cpu; default selects CUDA when available.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Restart a managed tokenizer output from epoch zero.",
    )
    return parser


def _repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def run(config: TokenizerTrainingConfig, *, device: str | None, force: bool) -> int:
    config = replace(
        config,
        embeddings_path=_repository_path(config.embeddings_path),
        item_texts_path=_repository_path(config.item_texts_path),
        output_dir=_repository_path(config.output_dir),
    )
    dataset = load_tokenizer_dataset(config.embeddings_path, config.item_texts_path)

    def report(metrics: dict[str, float]) -> None:
        print(
            f"Epoch {int(metrics['epoch']):03d}/{config.max_epochs}: "
            f"total={metrics['total']:.6f} "
            f"reconstruction={metrics['reconstruction']:.6f} "
            f"vq={metrics['vq']:.6f} "
            f"gate={metrics['gate_balance']:.6f}",
            flush=True,
        )

    result = train_semantic_tokenizer(
        dataset,
        config,
        device=device,
        force=force,
        report=report,
    )
    if result.already_complete:
        print("Tokenizer output is already complete; no training was repeated.")
    print(f"Items: {result.item_count}")
    print(f"Epochs completed: {result.epochs_completed}")
    print(f"Semantic ID collision rate: {result.collision_rate:.6f}")
    print(f"Codebook utilization: {result.codebook_utilization:.6f}")
    print(f"Semantic IDs: {result.semantic_ids_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = TokenizerTrainingConfig.from_yaml(args.config)
        return run(config, device=args.device, force=args.force)
    except Exception as error:
        print(f"Tokenizer training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
