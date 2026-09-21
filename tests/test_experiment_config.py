import tempfile
import unittest
from pathlib import Path

from cdr_framework.experiment_config import (
    AmazonPreprocessingConfig,
    QwenEmbeddingConfig,
    RecommendationTrainingConfig,
    TokenizerTrainingConfig,
)


class AmazonPreprocessingConfigTests(unittest.TestCase):
    def test_repository_configs_cover_all_three_benchmark_pairs(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "amazon_sports_clothing.yaml": (
                "Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry", "sports_to_clothing"
            ),
            "amazon_phones_electronics.yaml": (
                "Cell_Phones_and_Accessories", "Electronics", "phones_to_electronics"
            ),
            "amazon_books_movies.yaml": ("Books", "Movies_and_TV", "books_to_movies"),
        }
        for filename, (source, target, directory) in expected.items():
            path = root / "configs" / filename
            dataset = AmazonPreprocessingConfig.from_yaml(path)
            embedding = QwenEmbeddingConfig.from_yaml(path)
            tokenizer = TokenizerTrainingConfig.from_yaml(path)
            self.assertEqual((dataset.source_domain, dataset.target_domain), (source, target))
            self.assertEqual(dataset.processed_dir.name, directory)
            self.assertEqual(embedding.provider, "qwen3_local")
            self.assertEqual(embedding.output_dir.name, directory)
            self.assertEqual(tokenizer.output_dir.name, directory)

    def test_loads_expected_domains_and_converts_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "dataset:\n"
                "  source_domain: Sports_and_Outdoors\n"
                "  target_domain: Clothing_Shoes_and_Jewelry\n"
                "  raw_dir: data/raw/amazon2014\n"
                "  processed_dir: data/processed/sports_to_clothing\n"
                "  min_interactions_per_domain: 5\n",
                encoding="utf-8",
            )

            config = AmazonPreprocessingConfig.from_yaml(path)

            self.assertEqual(config.source_domain, "Sports_and_Outdoors")
            self.assertEqual(config.target_domain, "Clothing_Shoes_and_Jewelry")
            self.assertEqual(config.raw_dir, Path("data/raw/amazon2014"))
            self.assertEqual(
                config.processed_dir,
                Path("data/processed/sports_to_clothing"),
            )
            self.assertEqual(config.min_interactions_per_domain, 5)

    def test_accepts_all_three_gencdr_domain_pairs(self):
        pairs = (
            ("Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"),
            ("Cell_Phones_and_Accessories", "Electronics"),
            ("Books", "Movies_and_TV"),
        )
        for source, target in pairs:
            config = AmazonPreprocessingConfig(source_domain=source, target_domain=target)
            self.assertEqual((config.source_domain, config.target_domain), (source, target))

    def test_rejects_domain_pair_outside_the_benchmark_protocol(self):
        with self.assertRaises(ValueError):
            AmazonPreprocessingConfig(source_domain="Books", target_domain="Electronics")

    def test_rejects_threshold_below_three(self):
        with self.assertRaises(ValueError):
            AmazonPreprocessingConfig(min_interactions_per_domain=2)

    def test_rejects_non_positive_text_limit(self):
        with self.assertRaises(ValueError):
            AmazonPreprocessingConfig(max_text_chars=0)

    def test_loads_qwen_embedding_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "embedding:\n"
                "  provider: qwen\n"
                "  model: text-embedding-v4\n"
                "  dimension: 768\n"
                "  batch_size: 10\n",
                encoding="utf-8",
            )

            config = QwenEmbeddingConfig.from_yaml(path)

            self.assertEqual(config.model, "text-embedding-v4")
            self.assertEqual(config.dimension, 768)
            self.assertEqual(config.batch_size, 10)

    def test_loads_local_qwen3_8b_embedding_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "embedding:\n"
                "  provider: qwen3_local\n"
                "  model: Qwen/Qwen3-Embedding-8B\n"
                "  dimension: 768\n"
                "  batch_size: 1\n"
                "  device: cuda\n"
                "  max_sequence_length: 8192\n"
                "  model_path_env: QWEN3_EMBEDDING_MODEL_PATH\n",
                encoding="utf-8",
            )

            config = QwenEmbeddingConfig.from_yaml(path)

            self.assertEqual(config.provider, "qwen3_local")
            self.assertEqual(config.model, "Qwen/Qwen3-Embedding-8B")
            self.assertEqual(config.dimension, 768)
            self.assertEqual(config.batch_size, 1)
            self.assertEqual(config.device, "cuda")
            self.assertEqual(config.max_sequence_length, 8192)
            self.assertEqual(config.model_path_env, "QWEN3_EMBEDDING_MODEL_PATH")

    def test_loads_tokenizer_training_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "tokenizer:\n"
                "  input_dim: 768\n"
                "  hidden_dim: 128\n"
                "  codebook_size: 512\n"
                "  token_length: 4\n",
                encoding="utf-8",
            )
            config = TokenizerTrainingConfig.from_yaml(path)
            self.assertEqual(config.input_dim, 768)
            self.assertEqual(config.hidden_dim, 128)
            self.assertEqual(config.codebook_size, 512)
            self.assertEqual(config.token_length, 4)

    def test_loads_formal_recommendation_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "recommendation:\n"
                "  hidden_dim: 128\n"
                "  max_epochs: 40\n"
                "  top_ks: [5, 10, 20]\n",
                encoding="utf-8",
            )
            config = RecommendationTrainingConfig.from_yaml(path)
            self.assertEqual(config.hidden_dim, 128)
            self.assertEqual(config.max_epochs, 40)
            self.assertEqual(config.top_ks, (5, 10, 20))

    def test_l4e_yaml_hr_selection_metric_loads(self):
        from cdr_framework.text_config import TextCDRConfig

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "configs" / "text_sports_clothing_l4e.yaml"
        self.assertTrue(path.exists(), f"missing {path}")
        config = TextCDRConfig.from_yaml(path)
        self.assertEqual(config.reasoning_steps, 5)
        self.assertEqual(config.selection_metric, "HR@10")
        self.assertIn("text_cdr_v3_l4e", str(config.output_dir))

    def test_v4_p0_config_uses_isolated_output_and_explicit_scoring(self):
        from cdr_framework.text_config import TextCDRConfig

        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "configs" / "text_sports_clothing_v4_p0.yaml"
        self.assertTrue(path.exists(), f"missing {path}")
        config = TextCDRConfig.from_yaml(path)
        self.assertIn("text_cdr_v4", str(config.output_dir))
        self.assertEqual(config.selection_metric, "NDCG@10")
        self.assertEqual(config.retrieval_loss_weight, 1.0)
        self.assertEqual(config.generation_loss_weight, 1.0)
        self.assertEqual(config.retrieval_score_weight, 1.0)
        self.assertEqual(config.generation_score_weight, 1.0)
        self.assertEqual(config.fusion_normalization, "log_softmax")
        self.assertTrue(config.prototype_enabled)
        self.assertTrue(config.cd_injector_enabled)
        self.assertTrue(config.sp_injector_enabled)


if __name__ == "__main__":
    unittest.main()
