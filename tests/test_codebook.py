import unittest

import torch

from cdr_framework.codebook import ItemTokenIndex, SemanticCodebook, beam_search


class SemanticCodebookTest(unittest.TestCase):
    def test_residual_quantization_returns_fixed_length_tokens_and_reconstruction(self):
        codebook = SemanticCodebook(
            embeddings=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ],
                dtype=torch.float32,
            )
        )

        tokens, reconstruction, loss = codebook.quantize(
            torch.tensor([[0.9, 0.8]], dtype=torch.float32),
            token_length=2,
        )

        self.assertEqual(tokens.shape, (1, 2))
        self.assertEqual(reconstruction.shape, (1, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(loss.item(), 0.5)

    def test_item_token_index_maps_duplicate_sequences_with_tie_breaking(self):
        index = ItemTokenIndex()
        index.add("item-a", [3, 4], vector=torch.tensor([1.0, 0.0]))
        index.add("item-b", [3, 4], vector=torch.tensor([0.8, 0.2]))
        index.add("item-c", [9, 9], vector=torch.tensor([0.0, 1.0]))

        ranked = index.lookup_ranked(
            tokens=[3, 4],
            sequence_score=-0.2,
            query_vector=torch.tensor([1.0, 0.0]),
            top_k=2,
        )

        self.assertEqual([hit.item_id for hit in ranked], ["item-a", "item-b"])
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_beam_search_uses_length_normalized_completed_sequences(self):
        eos_id = 2

        def step(prefix):
            if prefix == [0]:
                return {1: -0.1, 3: -0.2}
            if prefix[-1] == 1:
                return {eos_id: -0.1}
            return {eos_id: -1.0}

        results = beam_search(
            step_log_probs=step,
            bos_id=0,
            eos_id=eos_id,
            beam_size=2,
            max_length=3,
            length_penalty=0.0,
        )

        self.assertEqual(results[0].tokens, [1])
        self.assertAlmostEqual(results[0].score, -0.2)


if __name__ == "__main__":
    unittest.main()
