import unittest

import torch

from cdr_framework.tokenization.residual_kmeans import (
    fit_residual_kmeans,
    quantize_residuals,
)


class ResidualKMeansTests(unittest.TestCase):
    def test_uses_an_independent_codebook_at_each_residual_level(self):
        vectors = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [3.0, 3.0],
                [3.0, 4.0],
                [4.0, 3.0],
                [4.0, 4.0],
            ],
            dtype=torch.float32,
        )

        tokens, reconstruction, codebooks = fit_residual_kmeans(
            vectors,
            codebook_size=4,
            token_length=2,
            iterations=8,
            seed=7,
        )

        self.assertEqual(tuple(tokens.shape), (8, 2))
        self.assertEqual(tuple(reconstruction.shape), (8, 2))
        self.assertEqual(tuple(codebooks.shape), (2, 4, 2))
        self.assertGreater(torch.unique(tokens[:, 0]).numel(), 1)
        self.assertGreater(torch.unique(tokens[:, 1]).numel(), 1)
        self.assertEqual(
            tokens.tolist(),
            quantize_residuals(vectors, codebooks)[0].tolist(),
        )

    def test_data_driven_quantization_avoids_sequence_collapse(self):
        generator = torch.Generator().manual_seed(11)
        vectors = torch.randn(64, 6, generator=generator)

        tokens, _, _ = fit_residual_kmeans(
            vectors,
            codebook_size=16,
            token_length=3,
            iterations=10,
            seed=11,
        )

        unique_sequences = len({tuple(row) for row in tokens.tolist()})
        collision_rate = 1.0 - unique_sequences / len(tokens)
        self.assertLessEqual(collision_rate, 0.10)
        for level in range(tokens.shape[1]):
            self.assertGreaterEqual(torch.unique(tokens[:, level]).numel(), 8)


if __name__ == "__main__":
    unittest.main()
