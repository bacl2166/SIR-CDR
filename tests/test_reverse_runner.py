import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from experiments.prepare_reverse_assets import parallel_cpu, ensure_gpu_idle, tokenizer_config


class ReverseRunnerTests(unittest.TestCase):
    def test_parallel_cpu_dispatches_independent_tasks_without_gpu_calls(self):
        tasks=['clothing_to_sports','electronics_to_phones','douban_movie_book']
        with tempfile.TemporaryDirectory() as folder, patch(
                'experiments.prepare_reverse_assets.launch',return_value=0) as launch:
            parallel_cpu(tasks,3,Path(folder),cpu_tokenizer=True)
        self.assertEqual({c.args[0] for c in launch.call_args_list},set(tasks))
        self.assertTrue(all(c.args[1]=='cpu' for c in launch.call_args_list))

    def test_busy_gpu_does_not_start_asset_work(self):
        with patch('experiments.prepare_reverse_assets._forward_workers',return_value=['12345']), patch(
                'experiments.prepare_reverse_assets.subprocess.check_output') as query:
            with self.assertRaisesRegex(RuntimeError,'Forward'):
                ensure_gpu_idle()
            query.assert_not_called()

    def test_tokenizer_isolated_and_explicitly_reversed(self):
        for pair in ('clothing_to_sports','electronics_to_phones','douban_movie_book'):
            cfg=tokenizer_config(pair,Path('/workspace'))
            self.assertEqual(cfg.output_dir.name,pair)
            self.assertEqual(cfg.input_dim,768)
            self.assertEqual(cfg.hidden_dim,128)


if __name__ == '__main__': unittest.main()
