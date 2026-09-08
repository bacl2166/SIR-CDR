import unittest
import math

from cdr_framework.config import DatasetConfig, FrameworkConfig, ModelConfig
from cdr_framework.metrics import coverage, hit_rate_at_k, mrr_at_k, ndcg_at_k, token_lookup_hit_rate


class ConfigMetricsTest(unittest.TestCase):
    def test_framework_config_defaults_match_refined_design(self):
        config = FrameworkConfig()

        self.assertEqual(config.dataset.pairs[0], ("amazon", "Sports", "Clothing"))
        self.assertIn(("douban", "Books", "Movies"), config.dataset.pairs)
        self.assertEqual(config.model.codebook_size, 512)
        self.assertEqual(config.model.token_length, 4)
        self.assertEqual(config.model.reasoning_steps, 3)
        self.assertFalse(config.embeddings.enable_api_calls)
        self.assertEqual(config.embeddings.provider_order[:2], ("qwen", "deepseek"))

    def test_metric_functions_score_ranked_recommendations(self):
        ranked = [[3, 7, 9], [2, 4, 5], [8, 1, 6]]
        positives = [7, 2, 0]

        self.assertAlmostEqual(hit_rate_at_k(ranked, positives, 2), 2 / 3)
        self.assertAlmostEqual(ndcg_at_k(ranked, positives, 2), (1 / math.log2(3) + 1.0) / 3)
        self.assertAlmostEqual(mrr_at_k(ranked, positives, 3), (1 / 2.0 + 1.0) / 3)
        self.assertAlmostEqual(coverage(ranked, total_items=10), 0.9)
        self.assertAlmostEqual(token_lookup_hit_rate([True, False, True]), 2 / 3)

    def test_config_rejects_invalid_model_settings(self):
        with self.assertRaises(ValueError):
            ModelConfig(codebook_size=0)

        with self.assertRaises(ValueError):
            DatasetConfig(pairs=())


if __name__ == "__main__":
    unittest.main()
