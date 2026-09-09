import unittest

import torch

from cdr_framework.codebook import SemanticCodebook as LegacySemanticCodebook
from cdr_framework.tokenization import DomainAdaptiveSemanticTokenizer, ItemTokenIndex, SemanticCodebook, beam_search


class TokenizationBoundaryTest(unittest.TestCase):
    def test_tokenization_package_reexports_existing_codebook_api(self):
        self.assertIs(SemanticCodebook, LegacySemanticCodebook)
        self.assertTrue(callable(beam_search))
        self.assertIsInstance(ItemTokenIndex(), ItemTokenIndex)

    def test_domain_adaptive_tokenizer_fuses_universal_and_domain_paths(self):
        torch.manual_seed(5)
        tokenizer = DomainAdaptiveSemanticTokenizer(input_dim=6, hidden_dim=4, num_domains=2)
        vectors = torch.randn(3, 6)
        domains = torch.tensor([0, 1, 0], dtype=torch.long)

        output = tokenizer(vectors, domains)

        self.assertEqual(output.final.shape, (3, 4))
        self.assertEqual(output.universal.shape, (3, 4))
        self.assertEqual(output.domain_specific.shape, (3, 4))
        self.assertEqual(output.gate.shape, (3, 1))
        self.assertTrue(torch.all(output.gate >= 0.0))
        self.assertTrue(torch.all(output.gate <= 1.0))


if __name__ == "__main__":
    unittest.main()
