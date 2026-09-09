import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from cdr_framework.experiment_config import TokenizerTrainingConfig
from cdr_framework.tokenization.training import (
    load_tokenizer_dataset,
    train_semantic_tokenizer,
)


class TokenizerTrainingTests(unittest.TestCase):
    def test_trains_exports_fixed_semantic_ids_and_reuses_completed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = torch.zeros((5, 6), dtype=torch.float32)
            embeddings[1:] = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                    [0.8, 0.2, 0.0, 0.0, 0.4, 0.1],
                    [0.0, 1.0, 0.0, 0.5, 0.0, 0.0],
                    [0.1, 0.8, 0.2, 0.4, 0.0, 0.0],
                ]
            )
            embeddings_path = root / "text_embeddings.pt"
            torch.save(embeddings, embeddings_path)
            item_texts_path = root / "item_texts.jsonl"
            with item_texts_path.open("w", encoding="utf-8") as stream:
                for item_id in range(1, 5):
                    domain = (
                        "Sports_and_Outdoors"
                        if item_id <= 2
                        else "Clothing_Shoes_and_Jewelry"
                    )
                    stream.write(
                        json.dumps(
                            {"item_id": item_id, "domain": domain, "text": f"item {item_id}"}
                        )
                        + "\n"
                    )
            config = TokenizerTrainingConfig(
                embeddings_path=embeddings_path,
                item_texts_path=item_texts_path,
                output_dir=root / "tokenizer",
                input_dim=6,
                hidden_dim=4,
                codebook_size=8,
                token_length=2,
                batch_size=2,
                max_epochs=2,
                seed=7,
            )
            dataset = load_tokenizer_dataset(embeddings_path, item_texts_path)

            first = train_semantic_tokenizer(dataset, config, device="cpu")

            tokens = torch.load(first.semantic_ids_path, map_location="cpu", weights_only=True)
            self.assertEqual(tuple(tokens.shape), (5, 2))
            self.assertTrue(torch.all(tokens[0] == -1))
            self.assertTrue(torch.all(tokens[1:] >= 0))
            self.assertTrue((config.output_dir / "tokenizer.pt").is_file())
            self.assertTrue((config.output_dir / "item_latents.pt").is_file())
            self.assertEqual(
                len((config.output_dir / "item_semantic_ids.jsonl").read_text().splitlines()),
                4,
            )

            second = train_semantic_tokenizer(dataset, config, device="cpu")
            self.assertTrue(second.already_complete)
            self.assertEqual(second.epochs_completed, 2)

    def test_rejects_embedding_dimension_mismatch(self):
        config = TokenizerTrainingConfig(
            input_dim=5,
            hidden_dim=4,
            codebook_size=8,
            token_length=2,
            batch_size=2,
            max_epochs=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings_path = root / "e.pt"
            texts_path = root / "i.jsonl"
            torch.save(torch.zeros((2, 6)), embeddings_path)
            texts_path.write_text(
                json.dumps(
                    {"item_id": 1, "domain": "Sports_and_Outdoors", "text": "x"}
                )
                + "\n"
            )
            dataset = load_tokenizer_dataset(embeddings_path, texts_path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                train_semantic_tokenizer(
                    dataset,
                    replace(config, output_dir=root / "out"),
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
