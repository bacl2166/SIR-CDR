import unittest

import torch

from cdr_framework.modules import (
    CrossDomainStructuralInjector,
    SpecificDomainStructuralInjector,
    UserImplicitReasoner,
)


class ModuleShapeTest(unittest.TestCase):
    def test_structural_injectors_pool_history_specific_signals(self):
        shared_tokens = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4) / 10.0
        cd_graph = torch.ones((6, 4), dtype=torch.float32)
        sp_tokens = shared_tokens + 1.0
        sp_graph = torch.full((6, 4), 2.0, dtype=torch.float32)
        source_history = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        target_history = torch.tensor([[3, 4], [4, 5]], dtype=torch.long)

        cd_injector = CrossDomainStructuralInjector(hidden_dim=4)
        sp_injector = SpecificDomainStructuralInjector(hidden_dim=4)

        cd_signal = cd_injector(shared_tokens, cd_graph, source_history, target_history)
        source_sp, target_sp = sp_injector(
            sp_tokens,
            sp_graph,
            sp_tokens,
            sp_graph,
            source_history,
            target_history,
        )

        self.assertEqual(cd_signal.shape, (2, 4))
        self.assertEqual(source_sp.shape, (2, 4))
        self.assertEqual(target_sp.shape, (2, 4))
        self.assertFalse(torch.allclose(source_sp, target_sp))

    def test_user_implicit_reasoner_returns_shared_and_private_heads(self):
        torch.manual_seed(1)
        reasoner = UserImplicitReasoner.random_init(hidden_dim=5, reasoning_steps=3, seed=9)

        shared, private, states = reasoner(
            context=torch.randn(2, 5),
            cd_signal=torch.randn(2, 5),
            target_sp_signal=torch.randn(2, 5),
        )

        self.assertEqual(shared.shape, (2, 5))
        self.assertEqual(private.shape, (2, 5))
        self.assertEqual(states.shape, (2, 3, 5))
        self.assertTrue(torch.all(torch.isfinite(states)))


if __name__ == "__main__":
    unittest.main()
