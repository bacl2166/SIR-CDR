from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from cdr_framework.datasets.amazon2014 import (  # noqa: E402
    AmazonCategoryFiles,
    category_files,
    download_category_files,
    iter_amazon_records,
)
from cdr_framework.datasets.amazon_preprocessing import (  # noqa: E402
    build_temporal_splits,
    prepare_domain_pair,
    write_preprocessed_artifacts,
)
from cdr_framework.experiment_config import AmazonPreprocessingConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-free Amazon Sports-to-Clothing artifacts."
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require existing raw gzip files instead of downloading them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing processed output after successful regeneration.",
    )
    return parser


def _repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _required_files(config: AmazonPreprocessingConfig) -> tuple[AmazonCategoryFiles, AmazonCategoryFiles]:
    source = category_files(config.source_domain, config.raw_dir)
    target = category_files(config.target_domain, config.raw_dir)
    return source, target


def _ensure_raw_files(
    config: AmazonPreprocessingConfig,
    skip_download: bool,
) -> tuple[AmazonCategoryFiles, AmazonCategoryFiles]:
    source, target = _required_files(config)
    missing = [
        path
        for path in (source.reviews, source.metadata, target.reviews, target.metadata)
        if not path.is_file()
    ]
    if skip_download and missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required raw files:\n{listing}")
    if not skip_download:
        source = download_category_files(config.source_domain, config.raw_dir)
        target = download_category_files(config.target_domain, config.raw_dir)
    return source, target


def run(config: AmazonPreprocessingConfig, *, skip_download: bool, force: bool) -> Path:
    config = replace(
        config,
        raw_dir=_repository_path(config.raw_dir),
        processed_dir=_repository_path(config.processed_dir),
    )
    if (config.processed_dir / "manifest.json").exists() and not force:
        raise FileExistsError(
            f"Processed output already exists: {config.processed_dir}. Use --force to replace it."
        )

    source_files, target_files = _ensure_raw_files(config, skip_download)
    prepared = prepare_domain_pair(
        iter_amazon_records(source_files.reviews),
        iter_amazon_records(target_files.reviews),
        iter_amazon_records(source_files.metadata),
        iter_amazon_records(target_files.metadata),
        min_interactions=config.min_interactions_per_domain,
        max_text_chars=config.max_text_chars,
    )
    samples = build_temporal_splits(prepared)
    output = write_preprocessed_artifacts(
        prepared,
        samples,
        config.processed_dir,
        source_domain=config.source_domain,
        target_domain=config.target_domain,
        min_interactions=config.min_interactions_per_domain,
        max_text_chars=config.max_text_chars,
        seed=config.seed,
        raw_files={
            "source_reviews": source_files.reviews,
            "source_metadata": source_files.metadata,
            "target_reviews": target_files.reviews,
            "target_metadata": target_files.metadata,
        },
        force=force,
    )
    print(f"Retained users: {prepared.statistics['retained_users']}")
    print(f"Source items: {prepared.statistics['retained_source_items']}")
    print(f"Target items: {prepared.statistics['retained_target_items']}")
    print(f"Train rows: {len(samples.train)}")
    print(f"Validation rows: {len(samples.validation)}")
    print(f"Test rows: {len(samples.test)}")
    print(f"Output: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = AmazonPreprocessingConfig.from_yaml(args.config)
        run(config, skip_download=args.skip_download, force=args.force)
    except Exception as error:
        print(f"Preprocessing failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
