import json
import tempfile
import unittest
from pathlib import Path

from experiments.prepare_gencdsr_assets import convert_dataset


def raw_fixture(root):
    mixed = [0, 0, 1, 1, 2, 2, 3, 3, 4]
    domains = [0, 1, 0, 1, 0, 1, 0, 1, 1]
    for domain, n in (("Book", 4), ("Movie", 5)):
        (root / domain).mkdir(parents=True)
        (root / domain / f"{domain}.item.json").write_text(json.dumps(
            {str(i): {"title": f"title{i}", "description": "中文文本"} for i in range(n)}))
        seq = [i for i, d in zip(mixed, domains) if d == (domain == "Movie")]
        (root / domain / f"{domain}.inter.json").write_text(json.dumps({"0": seq}))
    (root / "Book_Movie.inter.json").write_text(json.dumps({"0": mixed}))
    (root / "Book_Movie.domain.json").write_text(json.dumps({"0": domains}))


class AdapterTests(unittest.TestCase):
    def test_domain_namespaces_and_holdouts_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw_fixture(root)
            out = root / "output"
            result = convert_dataset(root, out, "Movie")
            mapping = json.loads((out / "mappings.json").read_text())
            self.assertEqual(len(mapping["id_to_item"]), 10)
            self.assertNotEqual(mapping["item_to_id"]["Book:0"], mapping["item_to_id"]["Movie:0"])
            train = [json.loads(s) for s in (out / "train.jsonl").read_text().splitlines()]
            valid = json.loads((out / "validation.jsonl").read_text())
            test = json.loads((out / "test.jsonl").read_text())
            self.assertEqual(valid["positive_target_item"], mapping["item_to_id"]["Movie:3"])
            self.assertEqual(test["positive_target_item"], mapping["item_to_id"]["Movie:4"])
            for row in train:
                self.assertNotIn(valid["positive_target_item"], row["target_items"])
                self.assertNotIn(test["positive_target_item"], row["target_items"])
                self.assertNotIn(mapping["item_to_id"]["Book:2"], row["source_items"])
            self.assertEqual(result["protocol"]["timestamp_kind"], "sequence_position_not_wall_clock")
            self.assertEqual(convert_dataset(root, out, "Movie"), result)
            (root / "Movie/Movie.item.json").write_text('{}')
            with self.assertRaises(ValueError):
                convert_dataset(root, out, "Movie")

    def test_projection_mismatch_and_empty_subsets_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw_fixture(root)
            (root / "Book/Book.inter.json").write_text('{"0": [3,2,1,0]}')
            with self.assertRaisesRegex(ValueError, "projection"):
                convert_dataset(root, root / "output", "Movie")

    def test_missing_metadata_keeps_catalog_with_disclosed_domain_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw_fixture(root)
            path = root / "Movie/Movie.item.json"
            records = json.loads(path.read_text()); records["0"] = {}
            path.write_text(json.dumps(records))
            report = convert_dataset(root, root / "out", "Movie")
            self.assertEqual(report["missing_metadata_domain_marker"], {"Movie": 1})

    def test_repeated_holdout_is_disclosed_and_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw_fixture(root)
            for name in ("Book_Movie.inter.json", "Book_Movie.domain.json", "Book/Book.inter.json", "Movie/Movie.inter.json"):
                p=root/name; data=json.loads(p.read_text()); data["1"]=list(data["0"])
                if name=="Book_Movie.inter.json": data["1"][-1]=1
                if name=="Movie/Movie.inter.json": data["1"][-1]=1
                p.write_text(json.dumps(data))
            report=convert_dataset(root,root/"out","Movie")
            self.assertEqual(report["excluded"]["repeated_eval_positive_user"],1)
            mapping=json.loads((root/"out/mappings.json").read_text())
            self.assertEqual(mapping["retained_original_users"],["0"])
