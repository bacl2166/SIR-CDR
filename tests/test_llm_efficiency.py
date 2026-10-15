"""Efficiency must not cache training features, leak labels or hide experiment changes."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_llm_training import config_at, offline_backbone
from cdr_framework import llm_training
from cdr_framework.llm_config import LLMCDRConfig
from cdr_framework.llm_efficiency import DeviceLossMeter
from cdr_framework.text_data import collate_text_rows
from cdr_framework.text_training import load_rows, evaluate as direct_evaluate


class LLMEfficiencyTests(unittest.TestCase):
    def test_device_loss_meter_weights_unequal_batches_and_detaches(self):
        meter = DeviceLossMeter()
        value = torch.tensor(2., requires_grad=True)
        meter.add({'total': value, 'aux': value * 3}, 3)
        meter.add({'total': value * 2, 'aux': value}, 1)
        self.assertFalse(meter.totals.requires_grad)
        self.assertIsNone(meter.totals.grad_fn)
        self.assertEqual(meter.means(), {'total': 2.5, 'aux': 5.0})
        self.assertIsNone(value.grad)

    def test_cached_validation_matches_direct_and_computes_catalog_once(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            config = replace(config_at(Path(d)), evaluation_batch_size=1)
            model, catalog = llm_training.build_model(config, torch.device('cpu'))
            rows = load_rows(config, catalog, 'validation')
            reference = direct_evaluate(model, rows, config, torch.device('cpu'))
            with patch.object(model.graph_encoder, 'forward', wraps=model.graph_encoder.forward) as graph:
                cached = llm_training.evaluate(model, rows, config, torch.device('cpu'))
                self.assertEqual(graph.call_count, 1)
            for key in reference:
                if key != 'seconds':
                    self.assertEqual(reference[key], cached[key], key)
            self.assertIsNone(model._evaluation_items)

    def test_cache_exception_cleanup_modes_and_training_gradients(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
            config = config_at(Path(d))
            model, catalog = llm_training.build_model(config, torch.device('cpu'))
            rows = load_rows(config, catalog, 'train')
            batch, seen = collate_text_rows(rows)
            with self.assertRaisesRegex(RuntimeError, 'eval'):
                with model.evaluation_catalog():
                    pass
            model.eval()
            with torch.no_grad():
                reference = {mode: model.rank(batch, seen, 5, mode).clone()
                             for mode in ('retrieval', 'hybrid', 'exhaustive', 'generate')}
                with model.evaluation_catalog():
                    for mode, ranked in reference.items():
                        self.assertTrue(torch.equal(ranked, model.rank(batch, seen, 5, mode)))
                    with self.assertRaisesRegex(RuntimeError, 'train'):
                        model.train()
                    with self.assertRaisesRegex(RuntimeError, 'load weights'):
                        model.load_state_dict(model.state_dict())
                    with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, 'autograd'):
                        model.item_states()
            with self.assertRaisesRegex(ValueError, 'deliberate'):
                with model.evaluation_catalog():
                    raise ValueError('deliberate')
            self.assertIsNone(model._evaluation_items)
            with model.evaluation_catalog():
                before = model.item_states()['base'].clone()
            with torch.no_grad():
                model.item_norm.bias.add_(0.25)
            with model.evaluation_catalog():
                self.assertFalse(torch.equal(before, model.item_states()['base']))
            model.train()
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                model(batch)['total'].backward()
                self.assertGreater(float(model.graph_encoder.projections['shared'].weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.reasoner.memory.write.weight.grad.abs().sum()), 0)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in model.named_parameters() if 'lora_' in n))

    def test_fast_presets_isolate_changes_and_sign_new_training_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        for task in ('phones_electronics', 'sports_clothing'):
            slow = LLMCDRConfig.from_yaml(root / f'configs/llm_{task}_v1.yaml')
            fast = LLMCDRConfig.from_yaml(root / f'configs/llm_{task}_qwen06_fast.yaml')
            self.assertEqual((fast.batch_size, fast.gradient_accumulation_steps), (4, 4))
            self.assertEqual(fast.evaluation_batch_size, 4)
            self.assertFalse(fast.llm_gradient_checkpointing)
            self.assertNotEqual(fast.output_dir, slow.output_dir)
            before, after = asdict(slow), asdict(fast)
            for key in ('batch_size', 'gradient_accumulation_steps', 'evaluation_batch_size',
                        'llm_gradient_checkpointing', 'output_dir'):
                before.pop(key); after.pop(key)
            self.assertEqual(before, after)
        with tempfile.TemporaryDirectory() as d:
            identity = llm_training.signature(config_at(Path(d)))
            self.assertIn('llm_model.py', identity['code'])
            self.assertIn('llm_efficiency.py', identity['code'])


if __name__ == '__main__':
    unittest.main()
