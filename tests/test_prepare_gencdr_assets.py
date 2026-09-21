import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.prepare_gencdr_assets import (
    ASSET_CONFIGS,
    processed_output_matches,
    tokenizer_output_ready,
)


class PrepareGenCDRAssetsTests(unittest.TestCase):
    def test_registry_uses_the_three_complete_experiment_configs(self):
        self.assertEqual(
            ASSET_CONFIGS,
            {
                "sports_to_clothing": Path("configs/amazon_sports_clothing.yaml"),
                "phones_to_electronics": Path("configs/amazon_phones_electronics.yaml"),
                "books_to_movies": Path("configs/amazon_books_movies.yaml"),
            },
        )

    def test_processed_manifest_must_match_configured_domains(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "manifest.json").write_text(
                json.dumps({"dataset": {
                    "source_domain": "Books",
                    "target_domain": "Movies_and_TV",
                }}),
                encoding="utf-8",
            )
            for name in ("mappings.json", "item_texts.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl"):
                (output / name).write_text("", encoding="utf-8")
            self.assertTrue(processed_output_matches(output, "Books", "Movies_and_TV"))
            with self.assertRaises(RuntimeError):
                processed_output_matches(output, "Books", "Electronics")

    def test_tokenizer_ready_requires_schema_and_passed_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertFalse(tokenizer_output_ready(output))
            torch.save({}, output / "item_latents.pt")
            torch.save({}, output / "semantic_ids.pt")
            torch.save({"schema_version": 2}, output / "tokenizer.pt")
            (output / "quality_report.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            self.assertTrue(tokenizer_output_ready(output))
            (output / "quality_report.json").write_text(
                json.dumps({"passed": False}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                tokenizer_output_ready(output)


if __name__ == "__main__":
    unittest.main()
