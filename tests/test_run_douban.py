import unittest
from unittest.mock import patch

from experiments.run_douban import commands, MODEL_CONFIG, ASSET_CONFIG
from cdr_framework.llm_config import LLMCDRConfig


class DoubanRunnerTests(unittest.TestCase):
    def test_independent_config_and_no_amazon_preprocessing(self):
        config = LLMCDRConfig.from_yaml(MODEL_CONFIG)
        self.assertEqual(config.batch_size * config.gradient_accumulation_steps, 16)
        self.assertEqual(config.reasoning_steps, 2)
        self.assertEqual(config.llm_model_name, 'Qwen/Qwen3-0.6B')
        self.assertEqual(config.processed_dir.name, 'douban_book_movie')
        self.assertFalse(config.llm_gradient_checkpointing)
        prepare = commands('prepare', 'cuda')
        self.assertEqual([cmd[cmd.index('--action')+1] for cmd in prepare],
                         ['embedding', 'tokenize', 'prepare', 'preflight'])
        self.assertNotIn('prepare_amazon.py', str(prepare))
        self.assertTrue(ASSET_CONFIG.is_file())

    def test_recommendation_uses_strict_snapshot_evaluation(self):
        planned = commands('run', 'cuda')
        self.assertEqual([cmd[cmd.index('--action')+1] for cmd in planned], ['train', 'all'])
        self.assertTrue(planned[-1][1].endswith('recover_llm_evaluation.py'))
        self.assertIn('--config', planned[-1])

    def test_tokenizer_receives_douban_domains_without_amazon_validation(self):
        from experiments.run_douban import tokenize
        with patch('experiments.train_tokenizer.run') as train:
            tokenize('cuda')
        self.assertEqual(train.call_args.kwargs,
                         {'device': 'cuda', 'force': False, 'source_domain': 'Book', 'target_domain': 'Movie'})
        self.assertEqual(train.call_args.args[0].output_dir.name, 'douban_book_movie')


if __name__ == '__main__':
    unittest.main()
