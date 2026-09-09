from __future__ import annotations

import ast
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import requests


AMAZON_2014_BASE_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
)


class AmazonRecordParseError(ValueError):
    """Raised when a line in an Amazon record file is not a dictionary."""


@dataclass(frozen=True)
class AmazonCategoryFiles:
    reviews: Path
    metadata: Path


def category_files(category: str, raw_dir: str | Path) -> AmazonCategoryFiles:
    root = Path(raw_dir)
    return AmazonCategoryFiles(
        reviews=root / f"reviews_{category}_5.json.gz",
        metadata=root / f"meta_{category}.json.gz",
    )


def iter_amazon_records(path: str | Path) -> Iterator[dict[str, Any]]:
    record_path = Path(path)
    with gzip.open(
        record_path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                try:
                    record = ast.literal_eval(stripped)
                except (SyntaxError, ValueError) as error:
                    raise AmazonRecordParseError(
                        f"Could not parse Amazon record at {record_path}, "
                        f"line {line_number}"
                    ) from error

            if not isinstance(record, dict):
                raise AmazonRecordParseError(
                    f"Amazon record at {record_path}, line {line_number} "
                    "must be a dictionary"
                )
            yield record


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_gzip(path: Path, chunk_size: int = 1024 * 1024) -> None:
    with gzip.open(path, "rb") as stream:
        while stream.read(chunk_size):
            pass


def _download_file(url: str, destination: Path) -> None:
    if destination.exists():
        try:
            _validate_gzip(destination)
        except (EOFError, OSError):
            pass
        else:
            return

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")
    offset = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}

    with requests.get(
        url,
        stream=True,
        headers=headers,
        timeout=(10, 120),
    ) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        with part_path.open("ab" if append else "wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)

    _validate_gzip(part_path)
    part_path.replace(destination)


def download_category_files(
    category: str,
    raw_dir: str | Path,
) -> AmazonCategoryFiles:
    files = category_files(category, raw_dir)
    _download_file(
        f"{AMAZON_2014_BASE_URL}/{files.reviews.name}",
        files.reviews,
    )
    _download_file(
        f"{AMAZON_2014_BASE_URL}/{files.metadata.name}",
        files.metadata,
    )
    return files
