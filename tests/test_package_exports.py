import unittest

import cdr_framework


class PackageExportsTest(unittest.TestCase):
    def test_top_level_exports_stable_framework_entry_points(self):
        expected_names = {
            "ContextPredictionFeedback",
            "DatasetConfig",
            "DualStructuralInjectionRecommender",
            "EmbeddingConfig",
            "FrameworkConfig",
            "ModelConfig",
            "TrainingConfig",
            "coverage",
            "datasets",
            "embeddings",
            "graphs",
            "hit_rate_at_k",
            "mrr_at_k",
            "ndcg_at_k",
            "token_lookup_hit_rate",
            "tokenization",
        }

        self.assertTrue(expected_names.issubset(set(cdr_framework.__all__)))
        for name in expected_names:
            self.assertTrue(hasattr(cdr_framework, name), name)


if __name__ == "__main__":
    unittest.main()
