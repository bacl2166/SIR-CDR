import json
import tempfile
import unittest
from pathlib import Path

from cdr_framework.datasets.amazon_preprocessing import (
    build_product_text,
    build_temporal_splits,
    prepare_domain_pair,
    write_preprocessed_artifacts,
)


def review(user, asin, timestamp):
    return {
        "reviewerID": user,
        "asin": asin,
        "unixReviewTime": timestamp,
        "reviewText": "must not become an item feature",
    }


def metadata(asin, title=None, **fields):
    record = {"asin": asin, **fields}
    if title is not None:
        record["title"] = title
    return record


class AmazonPreprocessingTests(unittest.TestCase):
    def test_build_product_text_cleans_and_orders_allowed_fields(self):
        record = metadata(
            "RAW-ID",
            title="  Trail &amp; Road <b>Shoe</b> ",
            brand="Acme",
            categories=[["Sports", "Running"]],
            feature=["Lightweight", ["Water resistant"]],
            description=["Fast <i>training</i>", "shoe"],
            reviewText="private review",
        )

        text = build_product_text(record, max_chars=1000)

        self.assertEqual(
            text,
            "Title: Trail & Road Shoe. Brand: Acme. "
            "Categories: Sports > Running. "
            "Features: Lightweight; Water resistant. "
            "Description: Fast training; shoe.",
        )
        self.assertNotIn("RAW-ID", text)
        self.assertNotIn("private review", text)

    def test_build_product_text_returns_empty_for_unusable_metadata(self):
        self.assertEqual(build_product_text({"asin": "ONLY-ID"}, 100), "")

    def test_prepare_pair_filters_after_missing_metadata_and_builds_mappings(self):
        source_reviews = [
            review("shared-user", "S2", 20),
            review("shared-user", "S1", 10),
            review("shared-user", "S3", 30),
            review("drops-after-metadata", "SX1", 1),
            review("drops-after-metadata", "SX2", 2),
            review("drops-after-metadata", "SX3", 3),
            review("source-only", "SO1", 1),
            review("source-only", "SO2", 2),
            review("source-only", "SO3", 3),
        ]
        target_reviews = [
            review("shared-user", "T1", 15),
            review("shared-user", "T3", 35),
            review("shared-user", "T2", 25),
            review("drops-after-metadata", "TX1", 1),
            review("drops-after-metadata", "TX2", 2),
            review("drops-after-metadata", "TX3", 3),
        ]
        source_metadata = [
            metadata("S1", "Source one"),
            metadata("S2", "Source two"),
            metadata("S3", "Source three"),
            metadata("SX1", "Drop one"),
            metadata("SX2", "Drop two"),
            metadata("SX3"),
            metadata("SO1", "Only one"),
            metadata("SO2", "Only two"),
            metadata("SO3", "Only three"),
        ]
        target_metadata = [
            metadata("T1", "Target one"),
            metadata("T2", "Target two"),
            metadata("T3", "Target three"),
            metadata("TX1", "Drop target one"),
            metadata("TX2", "Drop target two"),
            metadata("TX3", "Drop target three"),
        ]

        prepared = prepare_domain_pair(
            source_reviews,
            target_reviews,
            source_metadata,
            target_metadata,
            min_interactions=3,
            max_text_chars=12000,
        )

        self.assertEqual(set(prepared.user_to_id), {"shared-user"})
        self.assertEqual(prepared.item_to_id["Sports_and_Outdoors:S1"], 1)
        self.assertIsNone(prepared.id_to_item[0])
        self.assertEqual(
            [event.asin for event in prepared.source_events["shared-user"]],
            ["S1", "S2", "S3"],
        )
        self.assertEqual(
            [event.asin for event in prepared.target_events["shared-user"]],
            ["T1", "T2", "T3"],
        )
        self.assertTrue(prepared.item_texts[1].startswith("Title: Source one"))
        self.assertNotIn("reviewText", prepared.item_texts[1])
        self.assertEqual(prepared.statistics["retained_users"], 1)
        self.assertEqual(prepared.statistics["retained_source_interactions"], 3)
        self.assertEqual(prepared.statistics["retained_target_interactions"], 3)

    def test_temporal_splits_exclude_holdouts_and_future_source_events(self):
        source_reviews = [
            review("u1", f"S{index}", timestamp)
            for index, timestamp in enumerate((10, 25, 35, 55, 70), start=1)
        ]
        target_reviews = [
            review("u1", f"T{index}", timestamp)
            for index, timestamp in enumerate((20, 30, 40, 50, 60), start=1)
        ]
        prepared = prepare_domain_pair(
            source_reviews,
            target_reviews,
            [metadata(f"S{i}", f"Source {i}") for i in range(1, 6)],
            [metadata(f"T{i}", f"Target {i}") for i in range(1, 6)],
            min_interactions=5,
        )

        splits = build_temporal_splits(prepared)

        self.assertEqual(splits.validation[0].timestamp, 50)
        self.assertEqual(splits.test[0].timestamp, 60)
        source_id_at_55 = prepared.item_to_id["Sports_and_Outdoors:S4"]
        self.assertNotIn(source_id_at_55, splits.validation[0].source_items)
        self.assertIn(source_id_at_55, splits.test[0].source_items)
        held_out = {
            prepared.item_to_id["Clothing_Shoes_and_Jewelry:T4"],
            prepared.item_to_id["Clothing_Shoes_and_Jewelry:T5"],
        }
        self.assertTrue(
            held_out.isdisjoint(sample.positive_target_item for sample in splits.train)
        )
        for sample in (*splits.train, *splits.validation, *splits.test):
            self.assertNotIn(sample.positive_target_item, sample.target_items)

    def test_artifacts_round_trip_and_require_force_for_replacement(self):
        prepared = prepare_domain_pair(
            [review("u1", f"S{i}", i) for i in range(1, 6)],
            [review("u1", f"T{i}", i + 10) for i in range(1, 6)],
            [metadata(f"S{i}", f"Source {i}") for i in range(1, 6)],
            [metadata(f"T{i}", f"Target {i}") for i in range(1, 6)],
            min_interactions=5,
        )
        splits = build_temporal_splits(prepared)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.gz"
            raw.write_bytes(b"raw fingerprint")
            output = root / "processed"
            write_preprocessed_artifacts(
                prepared,
                splits,
                output,
                raw_files={"source_reviews": raw},
            )

            expected = {
                "mappings.json",
                "item_texts.jsonl",
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
                "statistics.json",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertIn("sha256", manifest["raw_files"]["source_reviews"])
            with self.assertRaises(FileExistsError):
                write_preprocessed_artifacts(prepared, splits, output)

            write_preprocessed_artifacts(prepared, splits, output, force=True)
            with (output / "test.jsonl").open(encoding="utf-8") as stream:
                self.assertEqual(len([json.loads(line) for line in stream]), 1)


if __name__ == "__main__":
    unittest.main()
