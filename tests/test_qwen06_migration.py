"""Check the active small-model presets and real Qwen3 adapter geometry."""
from dataclasses import asdict, replace
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3Model

from cdr_framework.llm_config import LLMCDRConfig
from cdr_framework.llm_reasoner import QwenLatentReasoner
from cdr_framework.llm_training import compact_state, restore_state


ROOT = Path(__file__).resolve().parents[1]
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class Qwen06MigrationTests(unittest.TestCase):
    def test_active_presets_change_only_model_revision_and_output(self):
        for pair in ("sports_clothing", "phones_electronics"):
            with self.subTest(pair=pair):
                current = LLMCDRConfig.from_yaml(ROOT / "configs" / f"llm_{pair}_v1.yaml")
                self.assertEqual(current.llm_model_name, "Qwen/Qwen3-0.6B")
                self.assertEqual(current.llm_revision, REVISION)
                current.require_pinned_model()
                legacy_path = ROOT / "configs" / f"llm_{pair}_4b_v1.yaml"
                self.assertTrue(legacy_path.is_file(), "Keep an explicit 4B reproduction preset")
                legacy = LLMCDRConfig.from_yaml(legacy_path)
                self.assertNotEqual(current.output_dir, legacy.output_dir)
                self.assertIn("qwen3_0_6b", current.output_dir.parts)
                before, after = asdict(legacy), asdict(current)
                for key in ("llm_model_name", "llm_revision", "output_dir"):
                    before.pop(key)
                    after.pop(key)
                self.assertEqual(before, after, "Do not change assets, losses or batching during model migration")

    def test_small_qwen_gqa_projection_memory_and_adapter_gradients(self):
        # Scale the 0.6B geometry: query width is twice hidden width, Q:KV=2:1.
        # A one-layer random model tests the production loader without downloading weights.
        config = LLMCDRConfig(hidden_dim=8, llm_model_name="Qwen/Qwen3-0.6B",
                              llm_revision=REVISION, llm_dtype="float32", lora_rank=2,
                              lora_alpha=4, llm_gradient_checkpointing=True)

        def backbone(width):
            base = Qwen3Model(Qwen3Config(vocab_size=32, hidden_size=width,
                intermediate_size=3 * width, num_hidden_layers=1,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16))
            base.config._commit_hash = REVISION
            return base

        with patch("transformers.AutoModel.from_pretrained", return_value=backbone(32)) as loader:
            reasoner = QwenLatentReasoner.from_pretrained(config)
        self.assertEqual(loader.call_args.args[0], "Qwen/Qwen3-0.6B")
        self.assertEqual(loader.call_args.kwargs["revision"], REVISION)
        self.assertEqual(reasoner.project.out_features, 32)
        self.assertEqual(reasoner.memory.read.in_features, 32)
        passes = []
        handle = reasoner.backbone.register_forward_hook(
            lambda module, args, result: passes.append(result.last_hidden_state.shape[1]))
        reasoner.train()
        shared, private, states = reasoner(*[torch.randn(1, 8) for _ in range(3)])
        self.assertEqual(passes, [4, 5])
        self.assertEqual(states.shape, (1, 2, 8))
        handle.remove()
        (shared.square().mean() + private.square().mean()).backward()
        for parameter in (reasoner.project.weight, reasoner.memory.write.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for name, p in reasoner.backbone.named_parameters() if "lora_" in name))
        self.assertTrue(all(p.grad is None for name, p in reasoner.backbone.named_parameters()
                            if "lora_" not in name))
        # Compact-state helpers operate on the enclosing recommendation model.
        wrapper = torch.nn.Module()
        wrapper.reasoner = reasoner
        state = compact_state(wrapper)
        with patch("transformers.AutoModel.from_pretrained", return_value=backbone(40)):
            incompatible = QwenLatentReasoner.from_pretrained(replace(config, llm_gradient_checkpointing=False))
        other = torch.nn.Module()
        other.reasoner = incompatible
        with self.assertRaises(RuntimeError):
            restore_state(other, state)


if __name__ == "__main__":
    unittest.main()
