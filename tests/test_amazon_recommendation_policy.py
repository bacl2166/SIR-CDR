from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cdr_framework import llm_training
from experiments import amazon_recommendation_policy as policy
from experiments import recover_llm_evaluation as recovery
from test_llm_training import config_at, offline_backbone


class AmazonRecommendationPolicyTests(unittest.TestCase):
    def test_dual_config_preserves_original_ndcg_and_binds_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            base = config_at(Path(d))
            config = policy.AmazonConfig(**llm_training.config_payload(base))
            config = replace(config, selection_metric=policy.MEAN_METRIC)
            metrics = {"NDCG@5": .1, "NDCG@10": .2, "HR@5": .3, "HR@10": .4}
            old_evaluate = llm_training.evaluate
            with patch.object(llm_training, "evaluate", return_value=dict(metrics)):
                with policy.installed():
                    result = llm_training.evaluate(None, [], config, torch.device("cpu"))
                    self.assertEqual(result["NDCG@5"], .1)
                    self.assertEqual(result["NDCG@10"], .2)
                    self.assertAlmostEqual(result[policy.MEAN_METRIC], .15)
                    self.assertIn("amazon_recommendation_policy.py", llm_training.signature(config)["code"])
            self.assertIs(llm_training.evaluate, old_evaluate)

    def test_fusion_selection_uses_equal_ndcg_then_hr_and_preserves_all_settings(self):
        results = [
            {"setting": {"pool": 500, "weight": .5, "normalization": "none"},
             "metrics": {"N@5": .1, "N@10": .2, "R@5": .3, "R@10": .4}},
            {"setting": {"pool": 1000, "weight": 1., "normalization": "none"},
             "metrics": {"N@5": .15, "N@10": .18, "R@5": .2, "R@10": .3}},
        ]
        self.assertIs(policy.select_fusion(results), results[1])
        self.assertEqual(len(results), 2)
        with self.assertRaises(ValueError):
            policy.metric_key({"N@5": float("nan"), "N@10": .1, "R@5": .1, "R@10": .2})

    def test_real_cpu_training_cache_verification_and_locked_test(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            root = Path(d)
            config = policy.AmazonConfig(**llm_training.config_payload(config_at(root)))
            config = replace(config, selection_metric=policy.MEAN_METRIC,
                             batch_size=2, evaluation_batch_size=2)
            with policy.installed():
                result = llm_training.train(config, torch.device("cpu"))
                self.assertTrue(result["complete"])
                self.assertIn(policy.MEAN_METRIC, result["best_validation"])
                for stage in ("cache", "verify", "test"):
                    recovery.recover_stage(config, root / "evaluation", stage, torch.device("cpu"))
                search = json.loads((root / "evaluation/search.json").read_text())
                self.assertEqual(search["selection_policy"], policy.SELECTION_POLICY)
                test = config.output_dir / "test.json"
                before = test.read_bytes()
                recovery.recover_stage(config, root / "evaluation", "test", torch.device("cpu"))
                self.assertEqual(test.read_bytes(), before)
                with self.assertRaisesRegex(RuntimeError, "test"):
                    recovery.recover_stage(config, root / "evaluation", "cache", torch.device("cpu"))
