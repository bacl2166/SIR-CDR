from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_llm_training import config_at, offline_backbone
from cdr_framework import llm_training, text_data
from cdr_framework.llm_efficiency import training_batch_to_device
from cdr_framework.text_model import TextSIRCDR
from experiments.benchmark_llm_cdr import benchmark_once, require_benchmark_card, sample_indices


class DoubanEfficiencyTests(unittest.TestCase):
    def test_server_presets_match_shared_protocol_except_isolated_paths(self):
        from cdr_framework.llm_config import LLMCDRConfig
        from experiments.run_gencdr_shared import model_config
        root = Path(__file__).resolve().parents[1]
        for pair in ('douban_book_movie', 'douban_movie_book'):
            config = LLMCDRConfig.from_yaml(root / f'configs/server/{pair}_efficiency_v1.yaml')
            a = llm_training.config_payload(config)
            b = llm_training.config_payload(model_config(root, pair, 42, 'paper-reference-v2'))
            for key in ('processed_dir', 'tokenizer_dir', 'output_dir'):
                a.pop(key); b.pop(key)
            self.assertEqual(a, b)
            self.assertIn('douban-efficiency-v1', str(config.output_dir))
            self.assertNotIn('paper-reference-v2', str(config.output_dir))

    def test_train_loader_omits_seen_only_preserves_labels_histories_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = config_at(Path(directory))
            targets = torch.arange(4, 12)
            old = text_data.load_text_rows(config.processed_dir / 'train.jsonl', 2, 12, targets)
            new = text_data.load_training_rows(config, 12, targets)
            self.assertEqual(len(old), len(new))
            for a, b in zip(old, new):
                self.assertEqual((a.user_id, a.source_items, a.target_items, a.positive_target_item),
                                 (b.user_id, b.source_items, b.target_items, b.positive_target_item))
                self.assertFalse(b.seen_items)
            batch, seen = text_data.collate_text_rows(old)
            compact = text_data.collate_training_rows(new)
            for key, value in batch.__dict__.items():
                self.assertTrue(torch.equal(value, getattr(compact, key)))
            self.assertIn(4, seen[0])  # history before max_sequence_length remains masked
            with (config.processed_dir / 'train.jsonl').open('a') as f:
                f.write('\n{"source_items":[999],"target_items":[4],"positive_target_item":5}')
            with self.assertRaises(ValueError):
                text_data.load_training_rows(config, 12, targets)

    def test_training_collator_does_not_access_seen_set(self):
        class NoIteration:
            def __iter__(self):
                raise AssertionError('training must not copy seen histories')
        row = text_data.TextRow(1, (1,), (4,), 5, NoIteration())
        batch = text_data.collate_training_rows([row])
        self.assertEqual(batch.positive_items.tolist(), [5])

    def compare_sequence(self, device):
        with tempfile.TemporaryDirectory() as directory, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            config = replace(config_at(Path(directory)), dropout=0.)
            model, catalog = llm_training.build_model(config, device)
            rows = text_data.load_training_rows(config, len(catalog.item_latents), catalog.target_item_ids)
            batch = text_data.collate_training_rows(rows)
            optimized = training_batch_to_device(batch, device)
            self.assertEqual(optimized.source_lengths.device.type, 'cpu')
            self.assertEqual(optimized.target_lengths.device.type, 'cpu')
            self.assertEqual(optimized.positive_items.device.type, device.type)
            rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
            original_sequence = model.sequence
            model.sequence = lambda *args: TextSIRCDR.sequence(model, *args)
            losses = model(batch.to(device))
            losses['total'].backward()
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            model.sequence = original_sequence
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            after = model(optimized)
            after['total'].backward()
            for name in losses:
                torch.testing.assert_close(after[name], losses[name], rtol=1e-5, atol=1e-6)
            for name, parameter in model.named_parameters():
                if name in gradients:
                    torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-5, atol=1e-6)
            self.assertGreater(float(model.graph_encoder.projections['shared'].weight.grad.abs().sum()), 0)

    def test_losses_and_gradients_match_legacy_on_cpu(self):
        self.compare_sequence(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'server CUDA verification required')
    def test_losses_and_gradients_match_legacy_on_cuda(self):
        self.compare_sequence(torch.device('cuda:0'))

    def test_sampling_is_seeded_identical_across_microbatch_layouts_and_preserves_rng(self):
        before = torch.get_rng_state().clone()
        a = sample_indices(500, 96, 42)
        self.assertEqual(a, sample_indices(500, 96, 42))
        self.assertNotEqual(a, sample_indices(500, 96, 43))
        self.assertTrue(all(0 <= i < 500 for i in a))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_gpu_admission_rejects_other_jobs_and_accepts_only_own_context_after_start(self):
        card = {'index': 6, 'uuid': 'GPU-six', 'pids': [123],
                'memory_free_mib': 40000, 'utilization': 0}
        with self.assertRaises(RuntimeError):
            require_benchmark_card([card], 6)
        self.assertEqual(require_benchmark_card([card], 6, own_pid=123)['uuid'], 'GPU-six')
        with self.assertRaises(RuntimeError):
            require_benchmark_card([card | {'pids': [123, 456]}], 6, own_pid=123)

    def test_gpu_admission_rejects_unattributed_utilization_or_insufficient_memory(self):
        card = {'index': 6, 'uuid': 'GPU-six', 'pids': [],
                'utilization': 80, 'memory_free_mib': 40000}
        with self.assertRaises(RuntimeError):
            require_benchmark_card([card], 6)
        with self.assertRaises(RuntimeError):
            require_benchmark_card([card | {'utilization': 0, 'memory_free_mib': 100}], 6)

    def test_optimizer_trajectory_with_dropout_matches_legacy_pipeline(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            config = config_at(Path(directory))
            old_config = replace(config, output_dir=config.output_dir.parent / 'legacy')
            old_sequence = lambda self, *args: TextSIRCDR.sequence(self, *args)
            from cdr_framework.llm_model import LLMSIRCDR
            def legacy_load(cfg, num_items, targets):
                return text_data.load_text_rows(cfg.processed_dir / 'train.jsonl',
                                               cfg.max_sequence_length, num_items, targets)
            with patch.object(llm_training, 'load_training_rows', legacy_load), \
                    patch.object(llm_training, 'collate_training_rows',
                                 lambda rows: text_data.collate_text_rows(rows)[0]), \
                    patch.object(llm_training, 'training_batch_to_device', lambda batch, device: batch.to(device)), \
                    patch.object(LLMSIRCDR, 'sequence', old_sequence):
                llm_training.train(old_config, torch.device('cpu'))
            llm_training.train(config, torch.device('cpu'))
            old = torch.load(old_config.output_dir / 'last_checkpoint.pt', weights_only=True)
            new = torch.load(config.output_dir / 'last_checkpoint.pt', weights_only=True)
            self.assertEqual(old['stale'], new['stale'])
            self.assertEqual(old['best_score'], new['best_score'])
            self.assertEqual(old['scheduler'], new['scheduler'])
            for key in old['model']:
                self.assertTrue(torch.equal(old['model'][key], new['model'][key]), key)

    def test_optimized_benchmark_never_loads_validation_or_writes_training_output(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            config = config_at(Path(directory))
            with patch('experiments.benchmark_llm_cdr.load_training_rows', wraps=text_data.load_training_rows) as load:
                report = benchmark_once(config, 'cpu', warmup_updates=1, measure_updates=1,
                                        input_pipeline='optimized')
            load.assert_called_once()
            self.assertEqual(report['input_pipeline'], 'optimized')
            self.assertEqual(report['measured_examples'], 3)
            self.assertFalse(config.output_dir.exists())


if __name__ == '__main__':
    unittest.main()
