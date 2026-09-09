import json
import tempfile
import unittest
from pathlib import Path

import torch

from cdr_framework.formal_recommendation import (
    FixedCatalog,
    FormalBatch,
    SIRCDRRecommender,
    load_recommendation_rows,
)


class FormalRecommendationTests(unittest.TestCase):
    def test_loader_pads_histories_without_using_user_ids_as_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "user_id": 91,
                        "source_items": [1, 2, 3],
                        "target_items": [5],
                        "positive_target_item": 6,
                        "timestamp": 10,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_recommendation_rows(path, max_sequence_length=2)

            self.assertEqual(rows[0].source_items, (2, 3))
            self.assertEqual(rows[0].target_items, (5,))
            self.assertEqual(rows[0].positive_target_item, 6)

    def test_model_uses_fixed_semantic_ids_and_scores_full_catalog(self):
        torch.manual_seed(4)
        latents = torch.randn(9, 6)
        latents[0].zero_()
        semantic_ids = torch.randint(0, 8, (9, 3))
        semantic_ids[0].fill_(-1)
        codebooks = torch.randn(3, 8, 6)
        catalog = FixedCatalog(
            item_latents=latents,
            semantic_ids=semantic_ids,
            codebooks=codebooks,
            target_item_ids=torch.tensor([5, 6, 7, 8]),
        )
        model = SIRCDRRecommender(catalog, hidden_dim=6, reasoning_steps=2)
        batch = FormalBatch(
            source_items=torch.tensor([[1, 2], [2, 3]]),
            source_lengths=torch.tensor([2, 2]),
            target_items=torch.tensor([[5, 0], [6, 7]]),
            target_lengths=torch.tensor([1, 2]),
            positive_items=torch.tensor([6, 8]),
        )

        output = model(batch)
        ranked = model.rank_full_catalog(batch, top_k=3, candidate_chunk_size=2)

        self.assertEqual(tuple(output.logits.shape), (2, 4, 10))
        self.assertIn("generation", output.losses)
        self.assertIn("retrieval", output.losses)
        self.assertIn("cpf", output.losses)
        self.assertEqual(tuple(ranked.shape), (2, 3))
        self.assertTrue(torch.all(torch.isin(ranked, catalog.target_item_ids)))

    def test_ranking_does_not_read_the_held_out_positive_label(self):
        torch.manual_seed(8)
        latents = torch.randn(7, 4)
        latents[0].zero_()
        semantic_ids = torch.randint(0, 5, (7, 2))
        semantic_ids[0].fill_(-1)
        model = SIRCDRRecommender(
            FixedCatalog(
                latents,
                semantic_ids,
                torch.randn(2, 5, 4),
                torch.tensor([4, 5, 6]),
            ),
            hidden_dim=4,
        )
        batch = FormalBatch(
            source_items=torch.tensor([[1, 2]]),
            source_lengths=torch.tensor([2]),
            target_items=torch.tensor([[4]]),
            target_lengths=torch.tensor([1]),
            positive_items=torch.tensor([999]),
        )

        ranked = model.rank_full_catalog(batch, top_k=2, candidate_chunk_size=1)

        self.assertEqual(tuple(ranked.shape), (1, 2))
        self.assertNotIn(4, ranked[0].tolist())


if __name__ == "__main__":
    unittest.main()
