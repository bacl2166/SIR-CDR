from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_llm_training import config_at, offline_backbone
from cdr_framework import llm_training
from cdr_framework.text_training import load_rows
from experiments.diagnose_llm_scoring import original_batches, compare_scores, probe_model
from experiments.tune_llm_fusion import collect_scores
from cdr_framework.formal_training import _sha256


class ScoringDiagnosticTests(unittest.TestCase):
    def test_cli_child_records_script_hash_before_model_probe(self):
        from experiments import diagnose_llm_scoring as diagnostic
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = config_at(root)
            config.output_dir.mkdir()
            checkpoint = config.output_dir / 'best_model.pt'
            checkpoint.write_bytes(b'checkpoint-fixture')
            cache_root = root / 'cache'
            cache_root.mkdir()
            cache = cache_root / 'validation_scores.pt'
            torch.save({'complete': True, 'identity': {'checkpoint_sha256': _sha256(checkpoint),
                        'training_identity': {}}, 'scores': {}}, cache)
            parameters = root / 'parameters'
            parameters.mkdir()
            (parameters / 'verification.json').write_text(json.dumps(
                {'identity': {'cache_sha256': _sha256(cache)}}))
            args = SimpleNamespace(config='fixture', parameters=parameters, cache_root=cache_root,
                                   device='cpu', profile='native', output=root / 'diagnostic.json')
            with patch.object(diagnostic, 'absolute_config', return_value=config), patch.object(
                    diagnostic, 'validate_checkpoint', return_value=({'model': {}}, {})), patch.object(
                    diagnostic.llm_training, 'build_model', side_effect=RuntimeError('probe deliberately skipped')):
                diagnostic._child(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(report['diagnostic_sha256'], _sha256(Path(diagnostic.__file__)))
            self.assertEqual(report['status'], 'failed')
            self.assertIn('deliberately skipped', report['error'])

    def test_benchmark_cli_records_script_hash_without_training(self):
        from experiments import benchmark_llm_cdr as benchmark
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = config_at(root)
            output = root / 'benchmark.json'
            with patch('sys.argv', ['benchmark', '--config', 'fixture', '--device', 'cpu',
                                   '--batch-sizes', '1', '--output', str(output)]), patch.object(
                    benchmark, 'absolute_config', return_value=config), patch.object(
                    benchmark, 'check_cached_model'), patch.object(
                    benchmark, 'benchmark_once', return_value={'status': 'ok'}):
                benchmark.main()
            self.assertEqual(json.loads(output.read_text())['benchmark_sha256'],
                             _sha256(Path(benchmark.__file__)))

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
