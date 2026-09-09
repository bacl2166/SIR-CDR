import json
import tempfile
import unittest
from pathlib import Path

import torch

from cdr_framework.experiment_config import RecommendationTrainingConfig
from cdr_framework.formal_training import run_formal_training


class FormalTrainingTests(unittest.TestCase):
    def test_one_epoch_exports_checkpoint_and_full_catalog_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            tokenizer = root / "tokenizer"
            processed.mkdir()
            tokenizer.mkdir()
            mappings = {
                "target_domain": "Clothing_Shoes_and_Jewelry",
                "id_to_item": [
                    None,
                    {"domain": "Sports_and_Outdoors", "asin": "s1"},
                    {"domain": "Sports_and_Outdoors", "asin": "s2"},
                    {"domain": "Clothing_Shoes_and_Jewelry", "asin": "t1"},
                    {"domain": "Clothing_Shoes_and_Jewelry", "asin": "t2"},
                    {"domain": "Clothing_Shoes_and_Jewelry", "asin": "t3"},
                ],
            }
            (processed / "mappings.json").write_text(json.dumps(mappings), encoding="utf-8")
            rows = [
                {"user_id": 0, "source_items": [1], "target_items": [3], "positive_target_item": 4, "timestamp": 1},
                {"user_id": 1, "source_items": [2], "target_items": [4], "positive_target_item": 5, "timestamp": 2},
            ]
            for split in ("train", "validation", "test"):
                (processed / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            latents = torch.randn(6, 4)
            latents[0].zero_()
            semantic_ids = torch.randint(0, 4, (6, 2))
            semantic_ids[0].fill_(-1)
            torch.save(latents, tokenizer / "item_latents.pt")
            torch.save(semantic_ids, tokenizer / "semantic_ids.pt")
            torch.save(
                {"schema_version": 2, "codebooks": torch.randn(2, 4, 4)},
                tokenizer / "tokenizer.pt",
            )
            (tokenizer / "quality_report.json").write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            config = RecommendationTrainingConfig(
                processed_dir=processed,
                tokenizer_dir=tokenizer,
                output_dir=root / "run",
                hidden_dim=4,
                batch_size=2,
                evaluation_batch_size=2,
                max_epochs=1,
                evaluation_every=1,
                top_ks=(1, 2),
                rerank_candidates=3,
            )

            result = run_formal_training(config, device="cpu")

            self.assertTrue((config.output_dir / "best_model.pt").is_file())
            self.assertTrue((config.output_dir / "manifest.json").is_file())
            self.assertIn("NDCG@2", result.best_validation)

            from experiments.diagnose_recommender import diagnose
            diagnostics = diagnose(config, torch.device("cpu"))
            self.assertEqual(diagnostics["examples"], 2)
            self.assertEqual(diagnostics["split"], "validation")
            self.assertEqual(diagnostics["candidate_recall"], 1.0)
            self.assertEqual(diagnostics["generation_reranked"], {
                key: value for key, value in result.best_validation.items()
                if key.startswith(("HR@", "NDCG@", "MRR@"))
            })

            semantic_ids[1, 0] = (semantic_ids[1, 0] + 1) % 4
            torch.save(semantic_ids, tokenizer / "semantic_ids.pt")
            with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
                run_formal_training(config, device="cpu")


if __name__ == "__main__":
    unittest.main()
