import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cdr_framework.datasets.amazon2014 import category_files
from cdr_framework.datasets.gencdr_inputs import raw_amazon
from cdr_framework.datasets.gencdr_protocol import build_shared, sha
from experiments.audit_cross_domain_candidates import PAIRS, direction_statistics, read_domain
from experiments.prepare_amazon_expansion import prepare


def fixture(root):
    raw = root / "data/raw/amazon2014"
    raw.mkdir(parents=True)
    domains = ("Grocery_and_Gourmet_Food", "Home_and_Kitchen", "Beauty",
               "Electronics", "Cell_Phones_and_Accessories")
    for number, domain in enumerate(domains):
        files = category_files(domain, raw)
        with gzip.open(files.reviews, "wt", encoding="utf-8") as stream:
            for index in range(6):
                stream.write(json.dumps({"reviewerID": "u", "asin": f"item{index}",
                                         "unixReviewTime": index * 10 + number,
                                         "reviewText": "HELDOUT_REVIEW_SECRET"}) + "\n")
        with gzip.open(files.metadata, "wt", encoding="utf-8") as stream:
            for index in range(5):
                stream.write(json.dumps({"asin": f"item{index}", "title": f"title{index}",
                                         "brand": "brand", "reviewText": "META_SECRET",
                                         "overall": 5, "salesRank": {"leak": 1}}) + "\n")
    source, target = "Cell_Phones_and_Accessories", "Electronics"
    events, items, provenance = raw_amazon(root, source, target)
    reference = root / "data/processed/gencdr-shared-loo-v1/phones_to_electronics"
    build_shared(events, items, source, target, reference, provenance=provenance)
    report = {"schema": "cross-domain-candidate-audit-v1", "files": {}, "directions": {}}
    for name, (left, right) in PAIRS.items():
        grouped = {}
        for domain in (left, right):
            path = category_files(domain, raw).reviews
            grouped[domain], _ = read_domain(path)
            report["files"][domain] = {"sha256": sha(path)}
        report["directions"][name] = direction_statistics(grouped[left], grouped[right])
        reverse = "_to_".join(reversed(name.split("_to_")))
        report["directions"][reverse] = direction_statistics(grouped[right], grouped[left])
    audit = root / "audit.json"
    audit.write_text(json.dumps(report), encoding="utf-8")
    return reference, audit


class AmazonExpansionTests(unittest.TestCase):
    def test_four_directions_reuse_protocol_static_text_and_audit_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, audit = fixture(root)
            before = (reference / "shared_manifest.json").read_bytes()
            report = prepare(root, audit, download=False)
            self.assertEqual(len(report["directions"]), 4)
            rules = json.loads((reference / "protocol.json").read_text(encoding="utf-8"))
            rules.pop("identity")
            for pair, result in report["directions"].items():
                self.assertTrue(result["protocol_rules_equal"])
                path = root / "data/processed/gencdr-shared-loo-v1" / pair
                current = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
                current.pop("identity")
                self.assertEqual(rules, current)
                text = (path / "item_texts.jsonl").read_text(encoding="utf-8")
                self.assertIn("title0", text)
                self.assertIn("Catalog identifier", text)
                self.assertNotIn("SECRET", text)
                self.assertNotIn("salesRank", text)
            self.assertEqual(before, (reference / "shared_manifest.json").read_bytes())
            self.assertEqual(report["directions"], prepare(root, audit, download=False)["directions"])

    def test_changed_electronics_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, audit = fixture(root)
            path = category_files("Electronics", root / "data/raw/amazon2014").metadata
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "reference raw"):
                prepare(root, audit, download=False)

    def test_changed_audit_counts_are_rejected_before_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, audit = fixture(root)
            report = json.loads(audit.read_text(encoding="utf-8"))
            report["directions"]["food_to_kitchen"]["train_rows"] = 99
            audit.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "audit counts"):
                prepare(root, audit, download=False)
            self.assertFalse((root / "data/processed/gencdr-shared-loo-v1/food_to_kitchen").exists())

    def test_metadata_change_cannot_reuse_an_already_prepared_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, audit = fixture(root)
            prepare(root, audit, download=False)
            output = root / "data/processed/gencdr-shared-loo-v1/food_to_kitchen/shared_manifest.json"
            before = output.read_bytes()
            meta = category_files("Home_and_Kitchen", root / "data/raw/amazon2014").metadata
            with gzip.open(meta, "at", encoding="utf-8") as stream:
                stream.write(json.dumps({"asin": "item5", "title": "new static title"}) + "\n")
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                prepare(root, audit, download=False)
            self.assertEqual(before, output.read_bytes())

    def test_review_replacement_between_audit_and_conversion_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, audit = fixture(root)
            def replace_before_read(data_root, source, target):
                path = category_files(source, root / "data/raw/amazon2014").reviews
                path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()), mtime=0))
                return raw_amazon(data_root, source, target)
            with patch("experiments.prepare_amazon_expansion.raw_amazon", side_effect=replace_before_read):
                with self.assertRaisesRegex(RuntimeError, "audited review"):
                    prepare(root, audit, download=False)
            self.assertFalse((root / "data/processed/gencdr-shared-loo-v1/food_to_kitchen").exists())
