from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_llm_training import config_at, offline_backbone
from cdr_framework import llm_training
from experiments.benchmark_llm_cdr import benchmark_once, candidate_configs
from experiments.run_llm_fast import runtime_config, run_pair


class ThroughputTests(unittest.TestCase):
    def test_candidate_batches_keep_nominal_effective_batch_and_sign_changes(self):
        with tempfile.TemporaryDirectory() as d:
            config = replace(config_at(Path(d)), gradient_accumulation_steps=16)
            candidates = candidate_configs(config, [1, 4, 8, 16], 'both')
            self.assertEqual(len(candidates), 8)
            for candidate in candidates:
                self.assertEqual(candidate.batch_size * candidate.gradient_accumulation_steps, 16)
            with self.assertRaises(ValueError):
                candidate_configs(config, [3], 'off')

    def test_benchmark_uses_only_train_and_never_writes_checkpoint_or_changes_rng(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            config = replace(config_at(Path(d)), batch_size=2, gradient_accumulation_steps=2)
            original_rng = torch.get_rng_state().clone()
            with patch('experiments.benchmark_llm_cdr.load_rows', wraps=llm_training.text_training.load_rows) as load:
                report = benchmark_once(config, torch.device('cpu'), warmup_updates=1, measure_updates=2)
            self.assertEqual([call.args[2] for call in load.call_args_list], ['train'])
            self.assertEqual(report['measured_examples'], 8)
            self.assertEqual(report['effective_batch_size'], 4)
            self.assertGreater(report['train_samples_per_second'], 0)
            self.assertEqual(report['optimizer_updates_measured'], 2)
            self.assertFalse(config.output_dir.exists())
            self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
            self.assertIn('code', report['training_identity'])

    def test_runner_uses_unique_runtime_and_does_not_reenter_search_after_test(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = runtime_config('phones_to_electronics', root, batch_size=8,
                                    evaluation_batch_size=4, checkpointing=False)
            self.assertEqual((config.batch_size, config.gradient_accumulation_steps), (8, 2))
            self.assertIn('qwen3_0_6b_fast_b8_eb4_gc0', config.output_dir.parts)
            config.output_dir.mkdir(parents=True)
            (config.output_dir / 'test.json').write_text('{}')
            with patch('experiments.run_llm_fast._run') as invoke:
                run_pair(config, root / 'runtime.yaml', root / 'tuning')
            commands = [call.args[0] for call in invoke.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertIn('train', commands[0])
            self.assertIn('test', commands[1])
            self.assertTrue(any('tune_llm_parameters.py' in part for part in commands[1]))
            with self.assertRaises(ValueError):
                runtime_config('douban_book_movie', root)

    def test_runner_full_pipeline_verifies_before_test(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = runtime_config('sports_to_clothing', root)
            with patch('experiments.run_llm_fast._run') as invoke:
                run_pair(config, root / 'runtime.yaml', root / 'tuning')
            commands = [call.args[0] for call in invoke.call_args_list]
            actions = [cmd[cmd.index('--action') + 1] for cmd in commands]
            self.assertEqual(actions, ['train', 'all', 'sweep', 'verify', 'test'])

    def test_actual_fast_train_tune_verify_test_and_completed_reuse(self):
        from experiments.tune_llm_fusion import run as fusion
        from experiments.tune_llm_parameters import run as parameters
        with tempfile.TemporaryDirectory() as d, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            root = Path(d)
            config = replace(config_at(root), batch_size=2, gradient_accumulation_steps=2,
                             evaluation_batch_size=2, llm_gradient_checkpointing=False)
            tuning = root / 'tuning'
            def execute(command):
                action = command[command.index('--action') + 1]
                if command[1].endswith('run_llm_cdr.py'):
                    llm_training.train(config, torch.device('cpu'))
                elif command[1].endswith('tune_llm_fusion.py'):
                    fusion(config, torch.device('cpu'), tuning, action)
                else:
                    parameters(config, tuning, Path(str(tuning) + '_parameters'), action, torch.device('cpu'))
            with patch('experiments.run_llm_fast._run', side_effect=execute):
                run_pair(config, root / 'runtime.yaml', tuning, device='cpu')
                test_path = config.output_dir / 'test.json'
                original = test_path.read_bytes()
                result = json.loads(original)
                self.assertEqual(result['state'], 'complete')
                self.assertEqual(result['metrics']['examples'], 2)
                for key in ('R@5', 'R@10', 'N@5', 'N@10'):
                    self.assertTrue(0 <= result['metrics'][key] <= 1)
                run_pair(config, root / 'runtime.yaml', tuning, device='cpu')
                self.assertEqual(original, test_path.read_bytes())


if __name__ == '__main__':
    unittest.main()
