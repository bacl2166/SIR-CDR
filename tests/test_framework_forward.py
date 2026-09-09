import unittest

import numpy as np
import torch

from cdr_framework.data import GraphBatch, InteractionBatch, ItemFeatures
from cdr_framework.framework import DualStructuralInjectionRecommender


class FrameworkForwardTest(unittest.TestCase):
    def test_synthetic_batch_produces_losses_and_recommendations(self):
        rng = np.random.default_rng(7)
        torch.manual_seed(7)
        model = DualStructuralInjectionRecommender.random_init(
            num_items=6,
            txt_dim=4,
            img_dim=4,
            attr_dim=3,
            hidden_dim=8,
            codebook_size=8,
            token_length=3,
            seed=11,
        )
        features = ItemFeatures(
            item_ids=torch.arange(6),
            id_vectors=torch.tensor(rng.normal(size=(6, 8)), dtype=torch.float32),
            text_vectors=torch.tensor(rng.normal(size=(6, 4)), dtype=torch.float32),
            image_vectors=torch.tensor(rng.normal(size=(6, 4)), dtype=torch.float32),
            attr_vectors=torch.tensor(rng.normal(size=(6, 3)), dtype=torch.float32),
            domains=("S", "S", "S", "T", "T", "T"),
        )
        graph = GraphBatch(
            source_edges=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            target_edges=torch.tensor([[3, 4], [4, 5]], dtype=torch.long),
            cross_edges=torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.long),
            num_items=6,
        )
        batch = InteractionBatch(
            user_ids=torch.tensor([100, 101]),
            source_sequences=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            target_sequences=torch.tensor([[3, 4], [4, 5]], dtype=torch.long),
            positive_target_items=torch.tensor([5, 3], dtype=torch.long),
        )

        output = model.forward(batch=batch, item_features=features, graph=graph)

        self.assertEqual(output.prefix.shape, (2, model.prefix_length, model.hidden_dim))
        self.assertEqual(output.target_tokens.shape, (2, model.token_length))
        self.assertIn("total", output.losses)
        self.assertGreater(output.losses["total"].item(), 0.0)

        recs = model.recommend(
            batch=batch,
            item_features=features,
            graph=graph,
            top_k=2,
            beam_size=3,
        )

        self.assertEqual(len(recs), 2)
        self.assertEqual(len(recs[0]), 2)
        self.assertEqual(len(recs[1]), 2)
        self.assertTrue(all(hit.item_id in {3, 4, 5} for hit in recs[0]))


if __name__ == "__main__":
    unittest.main()
