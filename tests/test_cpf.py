import unittest

import torch

from cdr_framework.losses import context_alignment_loss, cpf_total_loss
from cdr_framework.modules_cpf import ContextPredictionFeedback


class CPFTest(unittest.TestCase):
    def test_context_prediction_feedback_returns_prediction_and_losses(self):
        torch.manual_seed(13)
        cpf = ContextPredictionFeedback(hidden_dim=5, target_dim=7)
        latent = torch.randn(4, 5)
        context = torch.randn(4, 5)
        target = torch.randn(4, 7)
        reasoning_states = torch.randn(4, 3, 5)

        output = cpf(latent, context, target, reasoning_states)

        self.assertEqual(output.prediction.shape, (4, 7))
        self.assertIn("pred", output.losses)
        self.assertIn("ctx", output.losses)
        self.assertIn("var", output.losses)
        self.assertIn("total", output.losses)
        self.assertGreaterEqual(output.losses["total"].item(), 0.0)

    def test_context_alignment_loss_is_low_for_identical_vectors(self):
        vectors = torch.randn(3, 4)

        same = context_alignment_loss(vectors, vectors)
        opposite = context_alignment_loss(vectors, -vectors)

        self.assertLess(same.item(), 1e-6)
        self.assertGreater(opposite.item(), 1.0)

    def test_cpf_total_loss_combines_named_terms(self):
        losses = cpf_total_loss(
            pred_loss=torch.tensor(2.0),
            ctx_loss=torch.tensor(3.0),
            var_loss=torch.tensor(4.0),
            lambda_ctx=0.5,
            lambda_var=0.25,
        )

        self.assertEqual(set(losses), {"pred", "ctx", "var", "total"})
        self.assertAlmostEqual(losses["total"].item(), 4.5)


if __name__ == "__main__":
    unittest.main()
