from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdr_framework.experiment_config import (  # noqa: E402
    AmazonPreprocessingConfig,
    QwenEmbeddingConfig,
    TokenizerTrainingConfig,
)
from experiments.run_gencdr_benchmarks import (  # noqa: E402
    BENCHMARKS,
    preflight_assets,
    repository_identity,
)


ASSET_CONFIGS = {
    "sports_to_clothing": Path("configs/amazon_sports_clothing.yaml"),
    "phones_to_electronics": Path("configs/amazon_phones_electronics.yaml"),
    "books_to_movies": Path("configs/amazon_books_movies.yaml"),
}

PROCESSED_FILES = (
    "manifest.json",
    "mappings.json",
    "item_texts.jsonl",
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
)
TOKENIZER_FILES = (
    "item_latents.pt",
    "semantic_ids.pt",
    "tokenizer.pt",
    "quality_report.json",
)


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def processed_output_matches(output: Path, source_domain: str, target_domain: str) -> bool:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest.get("dataset", {})
    actual = (dataset.get("source_domain"), dataset.get("target_domain"))
    expected = (source_domain, target_domain)
    if actual != expected:
        raise RuntimeError(
            f"Processed dataset identity mismatch at {output}: actual={actual}, expected={expected}"
        )
    missing = [name for name in PROCESSED_FILES if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Processed dataset is incomplete at {output}: missing {missing}")
    return True


def tokenizer_output_ready(output: Path) -> bool:
    present = [name for name in TOKENIZER_FILES if (output / name).is_file()]
    if not present:
        return False
    if len(present) != len(TOKENIZER_FILES):
        return False
    quality = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    if quality.get("passed") is not True:
        raise RuntimeError(f"Tokenizer quality gate failed at {output}")
    checkpoint = torch.load(output / "tokenizer.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 2:
        raise RuntimeError(f"Tokenizer schema must be 2 at {output}")
    return True


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def prepare_pair(
    name: str,
    *,
    device: str,
    skip_download: bool,
) -> None:
    config_path = ROOT / ASSET_CONFIGS[name]
    dataset = AmazonPreprocessingConfig.from_yaml(config_path)
    embedding = QwenEmbeddingConfig.from_yaml(config_path)
    tokenizer = TokenizerTrainingConfig.from_yaml(config_path)
    dataset = replace(
        dataset,
        raw_dir=_rooted(dataset.raw_dir),
        processed_dir=_rooted(dataset.processed_dir),
    )
    embedding = replace(
        embedding,
        input_path=_rooted(embedding.input_path),
        output_dir=_rooted(embedding.output_dir),
        device=device,
    )
    tokenizer = replace(
        tokenizer,
        embeddings_path=_rooted(tokenizer.embeddings_path),
        item_texts_path=_rooted(tokenizer.item_texts_path),
        output_dir=_rooted(tokenizer.output_dir),
    )

    print(f"=== PREPARE {name} ===", flush=True)
    if tokenizer_output_ready(tokenizer.output_dir):
        print(f"Complete tokenizer already exists: {tokenizer.output_dir}", flush=True)
        return

    if not processed_output_matches(
        dataset.processed_dir, dataset.source_domain, dataset.target_domain
    ):
        if dataset.processed_dir.exists() and any(dataset.processed_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to replace unmanaged processed directory: {dataset.processed_dir}"
            )
        command = [
            sys.executable,
            "experiments/prepare_amazon.py",
            "--config",
            str(config_path),
        ]
        if skip_download:
            command.append("--skip-download")
        _run(command)
    processed_output_matches(dataset.processed_dir, dataset.source_domain, dataset.target_domain)

    _run([
        sys.executable,
        "experiments/embed_items.py",
        "--config",
        str(config_path),
        "--device",
        device,
    ])
    _run([
        sys.executable,
        "experiments/train_tokenizer.py",
        "--config",
        str(config_path),
        "--device",
        device,
    ])
    if not tokenizer_output_ready(tokenizer.output_dir):
        raise RuntimeError(f"Tokenizer preparation did not complete: {tokenizer.output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare resumable data, Qwen3 embeddings, and Semantic IDs for GenCDR pairs."
    )
    parser.add_argument("--pairs", nargs="+", choices=tuple(ASSET_CONFIGS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing Amazon gzip files and fail if any are missing.",
    )
    args = parser.parse_args()
    names = args.pairs or list(ASSET_CONFIGS)
    try:
        repository_identity(ROOT)
        for name in names:
            prepare_pair(name, device=args.device, skip_download=args.skip_download)
        report = preflight_assets(ROOT, tuple(BENCHMARKS[name] for name in names))
        print(json.dumps({"ready": report}, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(f"Asset preparation failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
