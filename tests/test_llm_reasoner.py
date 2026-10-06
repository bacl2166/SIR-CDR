"""Offline integration: real Qwen3/PEFT, no downloads or 4B allocation."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import torch
from transformers import Qwen3Config, Qwen3Model
from peft import LoraConfig, TaskType, get_peft_model

from cdr_framework.llm_config import LLMCDRConfig
from cdr_framework.llm_reasoner import PersistentMemory, QwenLatentReasoner


def tiny_reasoner(config=None):
    config = config or LLMCDRConfig(hidden_dim=8, llm_dtype="float32", lora_rank=2,
                                   lora_alpha=4, llm_revision="a" * 40)
    base = Qwen3Model(Qwen3Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, attention_dropout=0.0))
    base = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
        target_modules=["q_proj", "v_proj"], task_type=TaskType.FEATURE_EXTRACTION))
    return QwenLatentReasoner(base, config)


class LLMReasonerTests(unittest.TestCase):
    def test_production_loader_freezes_base_and_checks_revision(self):
        config = LLMCDRConfig(hidden_dim=8, llm_dtype="float32", lora_rank=2,
                              lora_alpha=4, llm_revision="a" * 40)
        def fixed_base():
            base = Qwen3Model(Qwen3Config(vocab_size=32, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8))
            base.config._commit_hash = "a" * 40
            return base
        with patch("transformers.AutoModel.from_pretrained", return_value=fixed_base()):
            reasoner = QwenLatentReasoner.from_pretrained(config)
            params = dict(reasoner.backbone.named_parameters())
            self.assertTrue(any(p.requires_grad for n, p in params.items() if "lora_" in n))
            self.assertTrue(all(not p.requires_grad for n, p in params.items() if "lora_" not in n))
        with patch("transformers.AutoModel.from_pretrained", return_value=fixed_base()):
            with self.assertRaises(ValueError):
                QwenLatentReasoner.from_pretrained(replace(config, llm_revision="b" * 40))

    def test_memory_reads_prior_state_and_resets_outside_forward(self):
        cell = PersistentMemory(4)
        h = torch.tensor([[1., -1., 2., -2.]])
        zero = torch.zeros_like(h)
        for layer in (cell.read, cell.forget, cell.write):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        first, mem = cell(h, zero)
        self.assertTrue(torch.equal(first, h * .5))
        second, _ = cell(h, mem)
        self.assertFalse(torch.allclose(first, second))
        again, _ = cell(h, zero)
        self.assertTrue(torch.equal(first, again))

    def test_two_actual_lm_passes_recurrent_gradient_and_frozen_base(self):
        torch.manual_seed(5)
        reasoner = tiny_reasoner()
        recorded = []
        def capture(module, args, result):
            result.last_hidden_state.retain_grad()
            recorded.append(result.last_hidden_state)
        handle = reasoner.backbone.register_forward_hook(capture)
        inputs = [torch.randn(2, 8, requires_grad=True) for _ in range(3)]
        shared, private, states = reasoner(*inputs)
        self.assertEqual(states.shape, (2, 2, 8))
        self.assertEqual([x.shape[1] for x in recorded], [4, 5])
        (shared.square().mean() + private.square().mean()).backward()
        self.assertGreater(recorded[0].grad.abs().sum().item(), 0)
        self.assertGreater(reasoner.project.weight.grad.abs().sum().item(), 0)
        self.assertGreater(reasoner.memory.write.weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
            for n, p in reasoner.backbone.named_parameters() if "lora_" in n))
        self.assertTrue(all(p.grad is None for n, p in reasoner.backbone.named_parameters()
                            if "lora_" not in n))
        handle.remove()

    def test_batch_independence_and_no_memory_leak(self):
        reasoner = tiny_reasoner().eval()
        inputs = [torch.randn(2, 8) for _ in range(3)]
        with torch.no_grad():
            a = reasoner(*inputs)[0]
            reasoner(*[torch.randn(2, 8) for _ in range(3)])
            b = reasoner(*inputs)[0]
            singles = torch.cat([reasoner(*[x[i:i+1] for x in inputs])[0] for i in range(2)])
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.allclose(a, singles, atol=1e-6))

    def test_bfloat16_checkpointed_backbone_keeps_projection_gradients(self):
        reasoner = tiny_reasoner()
        reasoner.backbone.to(dtype=torch.bfloat16)
        reasoner.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        reasoner.train()
        shared, private, _ = reasoner(*[torch.randn(1, 8) for _ in range(3)])
        (shared.square().mean() + private.square().mean()).backward()
        grad = reasoner.project.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_single_pass_ablation_and_invalid_config(self):
        config = LLMCDRConfig(hidden_dim=8, reasoning_steps=1, llm_dtype="float32")
        reasoner = tiny_reasoner(config)
        self.assertEqual(reasoner(*[torch.randn(1, 8) for _ in range(3)])[2].shape, (1, 1, 8))
        for changes in ({"reasoning_steps": 0}, {"feedback_steps": 2},
                        {"reasoning_enabled": False}, {"label_smoothing": float("nan")},
                        {"batch_size": 1, "contrastive_alignment": True},
                        {"gradient_accumulation_steps": 0}, {"llm_dtype": "int4"}):
            with self.assertRaises(ValueError):
                replace(config, **changes)


if __name__ == "__main__":
    unittest.main()
