"""Targeted recovery checks: stable features, exact verification, unchanged training identity."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_llm_training import config_at, offline_backbone
from cdr_framework import llm_training
from cdr_framework.text_data import collate_text_rows
from cdr_framework.text_training import load_rows
from experiments.llm_snapshot_scoring import cpu_item_snapshot, stable_topk, collect_snapshot_scores
from experiments.recover_llm_evaluation import recover_stage


class SnapshotRecoveryTests(unittest.TestCase):
    def test_partial_training_cannot_consume_validation_or_test(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            root = Path(d); config = config_at(root)
            llm_training.train(config, torch.device('cpu'), max_epochs_this_run=1)
            self.assertTrue((config.output_dir / 'best_model.pt').exists())
            self.assertFalse((config.output_dir / 'manifest.json').exists())
            for stage in ('cache', 'test'):
                with self.assertRaisesRegex(RuntimeError, 'Completed training'):
                    recover_stage(config, root / 'stable', stage, torch.device('cpu'))
            self.assertFalse((config.output_dir / 'test.json').exists())

    def test_tie_policy_and_masks_do_not_round_or_use_labels(self):
        scores = torch.tensor([[2., 2., 2.000001, -torch.inf]])
        ids = torch.tensor([[8, 2, 9, 1]])
        self.assertEqual(stable_topk(scores, ids, 4).tolist(), [[9, 2, 8, 0]])

    def test_cpu_snapshot_is_repeatable_and_does_not_modify_parameters(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            config = config_at(Path(d))
            model, catalog = llm_training.build_model(config, torch.device('cpu'))
            before = {k: v.clone() for k, v in model.state_dict().items()}
            a, b = cpu_item_snapshot(model), cpu_item_snapshot(model)
            self.assertTrue(torch.equal(a['shared'], b['shared']))
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))
            rows = load_rows(config, catalog, 'validation')
            with patch.object(model, 'item_states', side_effect=AssertionError('must use snapshot')):
                scores = collect_snapshot_scores(model, rows, config, torch.device('cpu'), a)
            self.assertEqual(scores['positive'].tolist(), [r.positive_target_item for r in rows])
            self.assertIsNone(model._evaluation_items)

    def test_complete_recovery_reuses_model_and_test_is_immutable(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            root = Path(d)
            config = replace(config_at(root), batch_size=2, evaluation_batch_size=2)
            llm_training.train(config, torch.device('cpu'))
            checkpoint = (config.output_dir / 'best_model.pt').read_bytes()
            output = root / 'stable'
            for stage in ('cache', 'verify', 'test'):
                recover_stage(config, output, stage, torch.device('cpu'))
            verified = json.loads((output / 'verification.json').read_text())
            self.assertTrue(verified['raw_scores_equal'])
            self.assertTrue(verified['rankings_equal'])
            result = json.loads((config.output_dir / 'test.json').read_text())
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['metrics']['examples'], 2)
            self.assertIn('evaluation_policy', result)
            original = (config.output_dir / 'test.json').read_bytes()
            recover_stage(config, output, 'test', torch.device('cpu'))
            self.assertEqual(original, (config.output_dir / 'test.json').read_bytes())
            self.assertEqual(checkpoint, (config.output_dir / 'best_model.pt').read_bytes())
            with self.assertRaisesRegex(RuntimeError, 'test'):
                recover_stage(config, output, 'cache', torch.device('cpu'))

    def test_corrupt_cache_or_changed_scores_block_test(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            root = Path(d); config = config_at(root)
            llm_training.train(config, torch.device('cpu'))
            output = root / 'stable'
            recover_stage(config, output, 'cache', torch.device('cpu'))
            from experiments import recover_llm_evaluation as recovery
            original = recovery.collect_snapshot_scores
            def changed(*args, **kwargs):
                value = original(*args, **kwargs)
                value['retrieval'] = value['retrieval'] + 0.01
                return value
            with patch.object(recovery, 'collect_snapshot_scores', side_effect=changed):
                with self.assertRaisesRegex(RuntimeError, 'differ'):
                    recover_stage(config, output, 'verify', torch.device('cpu'))
            with self.assertRaisesRegex(RuntimeError, 'verif'):
                recover_stage(config, output, 'test', torch.device('cpu'))
            self.assertFalse((config.output_dir / 'test.json').exists())


if __name__ == '__main__':
    unittest.main()
