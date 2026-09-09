from __future__ import annotations

import html
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping

from cdr_framework.datasets.amazon2014 import sha256_file


SOURCE_DOMAIN = "Sports_and_Outdoors"
TARGET_DOMAIN = "Clothing_Shoes_and_Jewelry"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


@dataclass(frozen=True)
class AmazonEvent:
    user_id: str
    asin: str
    domain: str
    timestamp: int


@dataclass(frozen=True)
class PreparedDomainPair:
    user_to_id: dict[str, int]
    item_to_id: dict[str, int]
    id_to_item: list[dict[str, str] | None]
    item_texts: dict[int, str]
    source_events: dict[str, tuple[AmazonEvent, ...]]
    target_events: dict[str, tuple[AmazonEvent, ...]]
    statistics: dict[str, int]


@dataclass(frozen=True)
class RecommendationSample:
    user_id: int
    source_items: tuple[int, ...]
    target_items: tuple[int, ...]
    positive_target_item: int
    timestamp: int


@dataclass(frozen=True)
class TemporalSamples:
    train: tuple[RecommendationSample, ...]
    validation: tuple[RecommendationSample, ...]
    test: tuple[RecommendationSample, ...]
    statistics: dict[str, int]


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    extractor = _TextExtractor()
    extractor.feed(html.unescape(str(value)))
    extractor.close()
    return " ".join(" ".join(extractor.parts).split())


def _flatten(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        flattened: list[str] = []
        for entry in value:
            flattened.extend(_flatten(entry))
        return flattened
    cleaned = _clean_scalar(value)
    return [cleaned] if cleaned else []


def _categories(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return _clean_scalar(value)
    paths: list[str] = []
    for entry in value:
        parts = _flatten(entry)
        if parts:
            paths.append(" > ".join(parts))
    return "; ".join(paths)


def _sentence(label: str, value: str) -> str:
    value = value.rstrip()
    if value and value[-1] not in ".!?":
        value += "."
    return f"{label}: {value}"


def build_product_text(record: Mapping[str, Any], max_chars: int) -> str:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    fields = (
        ("Title", _clean_scalar(record.get("title"))),
        ("Brand", _clean_scalar(record.get("brand"))),
        ("Categories", _categories(record.get("categories"))),
        ("Features", "; ".join(_flatten(record.get("feature")))),
        ("Description", "; ".join(_flatten(record.get("description")))),
    )
    text = " ".join(_sentence(label, value) for label, value in fields if value)
    return text[:max_chars].rstrip()


def _read_events(
    records: Iterable[Mapping[str, Any]],
    domain: str,
) -> tuple[list[AmazonEvent], int]:
    events: list[AmazonEvent] = []
    rejected = 0
    for record in records:
        user = record.get("reviewerID")
        asin = record.get("asin")
        timestamp = record.get("unixReviewTime")
        if not user or not asin or timestamp is None:
            rejected += 1
            continue
        try:
            normalized_timestamp = int(timestamp)
        except (TypeError, ValueError):
            rejected += 1
            continue
        events.append(
            AmazonEvent(
                user_id=str(user),
                asin=str(asin),
                domain=domain,
                timestamp=normalized_timestamp,
            )
        )
    return events, rejected


def _metadata_texts(
    records: Iterable[Mapping[str, Any]],
    reviewed_asins: set[str],
    max_chars: int,
) -> dict[str, str]:
    texts: dict[str, str] = {}
    for record in records:
        asin_value = record.get("asin")
        if not asin_value:
            continue
        asin = str(asin_value)
        if asin not in reviewed_asins or asin in texts:
            continue
        text = build_product_text(record, max_chars)
        if text:
            texts[asin] = text
    return texts


def _group(events: Iterable[AmazonEvent]) -> dict[str, tuple[AmazonEvent, ...]]:
    grouped: dict[str, list[AmazonEvent]] = {}
    for event in events:
        grouped.setdefault(event.user_id, []).append(event)
    return {
        user: tuple(sorted(rows, key=lambda row: (row.timestamp, row.asin)))
        for user, rows in grouped.items()
    }


def prepare_domain_pair(
    source_reviews: Iterable[Mapping[str, Any]],
    target_reviews: Iterable[Mapping[str, Any]],
    source_metadata: Iterable[Mapping[str, Any]],
    target_metadata: Iterable[Mapping[str, Any]],
    min_interactions: int = 5,
    max_text_chars: int = 12000,
) -> PreparedDomainPair:
    if min_interactions < 3:
        raise ValueError("min_interactions must be at least 3")

    raw_source, rejected_source = _read_events(source_reviews, SOURCE_DOMAIN)
    raw_target, rejected_target = _read_events(target_reviews, TARGET_DOMAIN)
    source_text = _metadata_texts(
        source_metadata, {event.asin for event in raw_source}, max_text_chars
    )
    target_text = _metadata_texts(
        target_metadata, {event.asin for event in raw_target}, max_text_chars
    )

    valid_source = [event for event in raw_source if event.asin in source_text]
    valid_target = [event for event in raw_target if event.asin in target_text]
    source_by_user = _group(valid_source)
    target_by_user = _group(valid_target)
    retained_users = sorted(
        user
        for user in source_by_user.keys() & target_by_user.keys()
        if len(source_by_user[user]) >= min_interactions
        and len(target_by_user[user]) >= min_interactions
    )

    source_events = {user: source_by_user[user] for user in retained_users}
    target_events = {user: target_by_user[user] for user in retained_users}
    source_asins = sorted({event.asin for rows in source_events.values() for event in rows})
    target_asins = sorted({event.asin for rows in target_events.values() for event in rows})

    item_to_id: dict[str, int] = {}
    id_to_item: list[dict[str, str] | None] = [None]
    item_texts: dict[int, str] = {}
    for domain, asins, texts in (
        (SOURCE_DOMAIN, source_asins, source_text),
        (TARGET_DOMAIN, target_asins, target_text),
    ):
        for asin in asins:
            item_id = len(id_to_item)
            item_to_id[f"{domain}:{asin}"] = item_id
            id_to_item.append({"domain": domain, "asin": asin})
            item_texts[item_id] = texts[asin]

    statistics = {
        "raw_source_interactions": len(raw_source),
        "raw_target_interactions": len(raw_target),
        "rejected_source_records": rejected_source,
        "rejected_target_records": rejected_target,
        "missing_text_source_interactions": len(raw_source) - len(valid_source),
        "missing_text_target_interactions": len(raw_target) - len(valid_target),
        "retained_users": len(retained_users),
        "retained_source_items": len(source_asins),
        "retained_target_items": len(target_asins),
        "retained_source_interactions": sum(map(len, source_events.values())),
        "retained_target_interactions": sum(map(len, target_events.values())),
    }
    return PreparedDomainPair(
        user_to_id={user: index for index, user in enumerate(retained_users)},
        item_to_id=item_to_id,
        id_to_item=id_to_item,
        item_texts=item_texts,
        source_events=source_events,
        target_events=target_events,
        statistics=statistics,
    )


def _sample_for(
    prepared: PreparedDomainPair,
    user: str,
    target_index: int,
) -> RecommendationSample | None:
    label = prepared.target_events[user][target_index]
    source = tuple(
        prepared.item_to_id[f"{SOURCE_DOMAIN}:{event.asin}"]
        for event in prepared.source_events[user]
        if event.timestamp < label.timestamp
    )
    target = tuple(
        prepared.item_to_id[f"{TARGET_DOMAIN}:{event.asin}"]
        for event in prepared.target_events[user][:target_index]
    )
    if not source or not target:
        return None
    return RecommendationSample(
        user_id=prepared.user_to_id[user],
        source_items=source,
        target_items=target,
        positive_target_item=prepared.item_to_id[f"{TARGET_DOMAIN}:{label.asin}"],
        timestamp=label.timestamp,
    )


def build_temporal_splits(prepared: PreparedDomainPair) -> TemporalSamples:
    train: list[RecommendationSample] = []
    validation: list[RecommendationSample] = []
    test: list[RecommendationSample] = []
    skipped = {"skipped_train_samples": 0, "skipped_validation_samples": 0, "skipped_test_samples": 0}

    for user in sorted(prepared.user_to_id, key=prepared.user_to_id.get):
        targets = prepared.target_events[user]
        for index in range(max(0, len(targets) - 2)):
            sample = _sample_for(prepared, user, index)
            if sample is None:
                skipped["skipped_train_samples"] += 1
            else:
                train.append(sample)

        for index, destination, key in (
            (len(targets) - 2, validation, "skipped_validation_samples"),
            (len(targets) - 1, test, "skipped_test_samples"),
        ):
            sample = _sample_for(prepared, user, index)
            if sample is None:
                skipped[key] += 1
            else:
                destination.append(sample)

    return TemporalSamples(tuple(train), tuple(validation), tuple(test), skipped)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_preprocessed_artifacts(
    prepared: PreparedDomainPair,
    samples: TemporalSamples,
    output_dir: str | Path,
    *,
    source_domain: str = SOURCE_DOMAIN,
    target_domain: str = TARGET_DOMAIN,
    min_interactions: int = 5,
    max_text_chars: int = 12000,
    seed: int = 42,
    raw_files: Mapping[str, str | Path] | None = None,
    force: bool = False,
) -> Path:
    output = Path(output_dir)
    if (output / "manifest.json").exists() and not force:
        raise FileExistsError(f"Processed output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    backup = output.with_name(output.name + ".backup")
    try:
        mappings = {
            "padding_item_id": 0,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "user_to_id": prepared.user_to_id,
            "id_to_user": [
                user for user, _ in sorted(prepared.user_to_id.items(), key=lambda pair: pair[1])
            ],
            "item_to_id": prepared.item_to_id,
            "id_to_item": prepared.id_to_item,
        }
        _write_json(temp / "mappings.json", mappings)
        counts = {
            "item_texts": _write_jsonl(
                temp / "item_texts.jsonl",
                (
                    {
                        "item_id": item_id,
                        "domain": prepared.id_to_item[item_id]["domain"],
                        "text": text,
                    }
                    for item_id, text in sorted(prepared.item_texts.items())
                ),
            ),
            "train": _write_jsonl(temp / "train.jsonl", (asdict(row) for row in samples.train)),
            "validation": _write_jsonl(
                temp / "validation.jsonl", (asdict(row) for row in samples.validation)
            ),
            "test": _write_jsonl(temp / "test.jsonl", (asdict(row) for row in samples.test)),
        }
        statistics = {**prepared.statistics, **samples.statistics, **{f"{key}_rows": value for key, value in counts.items()}}
        _write_json(temp / "statistics.json", statistics)

        raw_manifest = {}
        for name, value in sorted((raw_files or {}).items()):
            path = Path(value)
            raw_manifest[name] = {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "source_domain": source_domain,
                "target_domain": target_domain,
                "min_interactions_per_domain": min_interactions,
                "max_text_chars": max_text_chars,
                "seed": seed,
            },
            "raw_files": raw_manifest,
            "output_counts": counts,
        }
        _write_json(temp / "manifest.json", manifest)

        for path in temp.iterdir():
            if path.suffix in {".json", ".jsonl"}:
                if path.suffix == ".json":
                    json.loads(path.read_text(encoding="utf-8"))
                else:
                    with path.open(encoding="utf-8") as stream:
                        for line in stream:
                            json.loads(line)

        if backup.exists():
            shutil.rmtree(backup)
        if output.exists():
            output.replace(backup)
        temp.replace(output)
        if backup.exists():
            shutil.rmtree(backup)
        return output
    except Exception:
        if not output.exists() and backup.exists():
            backup.replace(output)
        raise
    finally:
        if temp.exists():
            shutil.rmtree(temp)
