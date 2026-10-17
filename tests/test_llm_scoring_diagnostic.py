from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_llm_training import config_at, offline_backbone
from cdr_framework import llm_training
from cdr_framework.text_training import load_rows
from experiments.diagnose_llm_scoring import original_batches, compare_scores, probe_model
from experiments.tune_llm_fusion import collect_scores


class ScoringDiagnosticTests(unittest.TestCase):
    def test_original_batch_padding_and_order_are_preserved(self):
        self.assertEqual(original_batches(11, 4, [5, 1, 10], 3), list(range(11)))
        self.assertEqual(original_batches(20, 4, [9, 10, 14], 3), list(range(8, 16)))
        with self.assertRaises(ValueError):
            original_batches(10, 4, [10], 3)

    def test_score_comparison_aligns_item_ids_instead_of_candidate_positions(self):
        a = {'candidates': torch.tensor([[7, 9]]), 'retrieval': torch.tensor([[1., 2.]]),
             'generation': torch.tensor([[3., 4.]])}
        b = {key: value.flip(1) for key, value in a.items()}
        report = compare_scores(a, b)
        self.assertFalse(report['candidate_order_equal'])
        self.assertEqual(report['maximum_common_item_score_difference'], {'retrieval': 0., 'generation': 0.})

    def test_small_probe_localizes_repeated_scores_without_training_artifacts(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            config = config_at(Path(d))
            model, catalog = llm_training.build_model(config, torch.device('cpu'))
            rows = load_rows(config, catalog, 'validation')
            cache = collect_scores(model, rows, config, torch.device('cpu'), 1000)
            report = probe_model(model, rows, config, torch.device('cpu'), cache)
            self.assertTrue(report['repeated_scores']['bitwise_equal'])
            self.assertTrue(report['repeated_catalog']['all_equal'])
            self.assertTrue(report['fixed_catalog_encoding']['all_equal'])
            self.assertFalse(config.output_dir.exists())


if __name__ == '__main__':
    unittest.main()
