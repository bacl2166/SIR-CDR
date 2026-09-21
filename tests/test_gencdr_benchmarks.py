import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_gencdr_benchmarks import (
    BENCHMARKS,
    AssetPreflightError,
    action_requires_device,
    collect_metrics,
    preflight_assets,
    select_validation_setting,
    validate_repository_identity,
)


class GenCDRBenchmarkRunnerTests(unittest.TestCase):
    def test_only_model_actions_require_a_device(self):
        for action in ("train", "evaluate", "all"):
            self.assertTrue(action_requires_device(action))
        for action in ("preflight", "summarize"):
            self.assertFalse(action_requires_device(action))

    def test_repository_identity_requires_clean_matching_revision(self):
        identity = validate_repository_identity("abc", "abc", "")
        self.assertEqual(identity["commit"], "abc")
        with self.assertRaises(AssetPreflightError):
            validate_repository_identity("abc", "def", "")
        with self.assertRaises(AssetPreflightError):
            validate_repository_identity("abc", "abc", "M tracked.py")

    def test_registry_contains_the_three_paper_dataset_pairs(self):
        self.assertEqual(
            tuple(BENCHMARKS),
            ("sports_to_clothing", "phones_to_electronics", "books_to_movies"),
        )
        self.assertEqual(BENCHMARKS["sports_to_clothing"].target_domain, "Clothing_Shoes_and_Jewelry")
        self.assertEqual(BENCHMARKS["phones_to_electronics"].target_domain, "Electronics")
        self.assertEqual(BENCHMARKS["books_to_movies"].target_domain, "Movies_and_TV")

    def test_preflight_reports_all_missing_assets_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(AssetPreflightError) as raised:
                preflight_assets(root, tuple(BENCHMARKS.values()))
            message = str(raised.exception)
            for pair in BENCHMARKS:
                self.assertIn(pair, message)
            self.assertIn("train.jsonl", message)
            self.assertIn("tokenizer.pt", message)

    def test_validation_selection_uses_ndcg_then_lower_generation_weight(self):
        reports = [
            {
                "identity": {"fusion_normalization": "none"},
                "best_validation_setting": {
                    "NDCG@10": 0.20,
                    "generation_score_weight": 1.0,
                    "retrieval_score_weight": 1.0,
                },
            },
            {
                "identity": {"fusion_normalization": "zscore"},
                "best_validation_setting": {
                    "NDCG@10": 0.20,
                    "generation_score_weight": 0.5,
                    "retrieval_score_weight": 1.0,
                },
            },
            {
                "identity": {"fusion_normalization": "log_softmax"},
                "best_validation_setting": {
                    "NDCG@10": 0.19,
                    "generation_score_weight": 0.0,
                    "retrieval_score_weight": 1.0,
                },
            },
        ]
        selected = select_validation_setting(reports)
        self.assertEqual(selected["fusion_normalization"], "zscore")
        self.assertEqual(selected["generation_score_weight"], 0.5)

    def test_collect_metrics_writes_machine_readable_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = {
                "sports_to_clothing": {
                    "checkpoint_sha256": "abc",
                    "epoch": 3,
                    "metrics": {"HR@10": 0.1, "NDCG@10": 0.05, "MRR@10": 0.03},
                },
                "phones_to_electronics": {
                    "checkpoint_sha256": "def",
                    "epoch": 4,
                    "metrics": {"HR@10": 0.2, "NDCG@10": 0.10, "MRR@10": 0.06},
                },
                "books_to_movies": {
                    "checkpoint_sha256": "ghi",
                    "epoch": 5,
                    "metrics": {"HR@10": 0.3, "NDCG@10": 0.15, "MRR@10": 0.09},
                },
            }
            json_path, csv_path = collect_metrics(metrics, root, seed=42)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["datasets"]["books_to_movies"]["metrics"]["NDCG@10"], 0.15)
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("dataset,epoch,checkpoint_sha256", csv_text)
            self.assertIn("phones_to_electronics,4,def", csv_text)


if __name__ == "__main__":
    unittest.main()
