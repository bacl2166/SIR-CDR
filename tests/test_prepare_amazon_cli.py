import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "experiments" / "prepare_amazon.py"


def write_gzip_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


class PrepareAmazonCLITests(unittest.TestCase):
    def test_offline_cli_writes_manifest_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            processed = root / "processed"
            config = root / "config.yaml"
            config.write_text(
                "dataset:\n"
                "  source_domain: Sports_and_Outdoors\n"
                "  target_domain: Clothing_Shoes_and_Jewelry\n"
                f"  raw_dir: {raw.as_posix()}\n"
                f"  processed_dir: {processed.as_posix()}\n"
                "  min_interactions_per_domain: 5\n"
                "  max_text_chars: 12000\n"
                "  seed: 42\n",
                encoding="utf-8",
            )
            source_reviews = [
                {"reviewerID": "u1", "asin": f"S{i}", "unixReviewTime": i}
                for i in range(1, 6)
            ]
            target_reviews = [
                {"reviewerID": "u1", "asin": f"T{i}", "unixReviewTime": i + 10}
                for i in range(1, 6)
            ]
            source_meta = [{"asin": f"S{i}", "title": f"Source {i}"} for i in range(1, 6)]
            target_meta = [{"asin": f"T{i}", "title": f"Target {i}"} for i in range(1, 6)]
            write_gzip_records(raw / "reviews_Sports_and_Outdoors_5.json.gz", source_reviews)
            write_gzip_records(raw / "meta_Sports_and_Outdoors.json.gz", source_meta)
            write_gzip_records(
                raw / "reviews_Clothing_Shoes_and_Jewelry_5.json.gz", target_reviews
            )
            write_gzip_records(raw / "meta_Clothing_Shoes_and_Jewelry.json.gz", target_meta)

            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--config", str(config), "--skip-download"],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((processed / "manifest.json").is_file())
            for label in (
                "Retained users: 1",
                "Source items: 5",
                "Target items: 5",
                "Train rows:",
                "Validation rows: 1",
                "Test rows: 1",
            ):
                self.assertIn(label, result.stdout)

    def test_skip_download_lists_every_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(
                "dataset:\n"
                "  source_domain: Sports_and_Outdoors\n"
                "  target_domain: Clothing_Shoes_and_Jewelry\n"
                f"  raw_dir: {(root / 'missing').as_posix()}\n"
                f"  processed_dir: {(root / 'processed').as_posix()}\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--config", str(config), "--skip-download"],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reviews_Sports_and_Outdoors_5.json.gz", result.stderr)
            self.assertIn("meta_Sports_and_Outdoors.json.gz", result.stderr)
            self.assertIn("reviews_Clothing_Shoes_and_Jewelry_5.json.gz", result.stderr)
            self.assertIn("meta_Clothing_Shoes_and_Jewelry.json.gz", result.stderr)


if __name__ == "__main__":
    unittest.main()
