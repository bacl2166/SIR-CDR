from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cdr_framework.embeddings import (  # noqa: E402
    QwenTextEmbeddingProvider,
    load_item_texts,
    run_embedding_job,
    sha256_path,
)
from cdr_framework.experiment_config import QwenEmbeddingConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate resumable Qwen text-embedding-v4 item vectors."
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Embed one item without writing artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Restart a managed embedding output from the beginning.",
    )
    return parser


def _repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def run(config: QwenEmbeddingConfig, *, smoke_test: bool, force: bool) -> int:
    input_path = _repository_path(config.input_path)
    output_dir = _repository_path(config.output_dir)
    items = load_item_texts(input_path, limit=1 if smoke_test else None)
    provider = QwenTextEmbeddingProvider(
        api_key_env=config.api_key_env,
        base_url_env=config.base_url_env,
        model=config.model,
        dim=config.dimension,
    )
    if smoke_test:
        vectors = provider.encode_text([items[0].text])
        print("Qwen embedding smoke test succeeded")
        print(f"Model: {config.model}")
        print(f"Items: {vectors.shape[0]}")
        print(f"Dimension: {vectors.shape[1]}")
        return 0

    result = run_embedding_job(
        items,
        provider,
        output_dir,
        input_sha256=sha256_path(input_path),
        model=config.model,
        dimension=config.dimension,
        batch_size=config.batch_size,
        max_retries=config.max_retries,
        retry_initial_seconds=config.retry_initial_seconds,
        force=force,
    )
    print("Qwen item embedding completed")
    print(f"Items: {result.item_count}")
    print(f"API batches this run: {result.api_batches}")
    print(f"Resumed batches: {result.resumed_batches}")
    print(f"Embeddings: {result.embeddings_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = QwenEmbeddingConfig.from_yaml(args.config)
        return run(config, smoke_test=args.smoke_test, force=args.force)
    except Exception as error:
        print(f"Embedding failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
