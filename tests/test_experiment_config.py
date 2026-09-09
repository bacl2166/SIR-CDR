import tempfile
import unittest
from pathlib import Path

from cdr_framework.experiment_config import (
    AmazonPreprocessingConfig,
    QwenEmbeddingConfig,
    TokenizerTrainingConfig,
)


class AmazonPreprocessingConfigTests(unittest.TestCase):
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

    def test_rejects_unexpected_source_domain(self):
        with self.assertRaises(ValueError):
            AmazonPreprocessingConfig(source_domain="Books")

    def test_rejects_unexpected_target_domain(self):
        with self.assertRaises(ValueError):
            AmazonPreprocessingConfig(target_domain="Movies_and_TV")

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


if __name__ == "__main__":
    unittest.main()
