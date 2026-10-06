import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
import torch

from test_text_pipeline import fixture
from test_llm_reasoner import tiny_reasoner
from cdr_framework.llm_config import LLMCDRConfig
from cdr_framework import llm_training as run
from cdr_framework.text_data import collate_text_rows
from cdr_framework.text_training import load_rows


def config_at(root):
    values = asdict(fixture(root))
    values.update(reasoning_steps=2, feedback_steps=1, batch_size=1,
                  contrastive_alignment=False,
                  gradient_accumulation_steps=3, llm_dtype="float32",
                  llm_revision="a" * 40, lora_rank=2, lora_alpha=4)
    return LLMCDRConfig(**values)


def offline_backbone(config):
    # Stand in for downloading fixed pretrained weights; the model itself is real.
    with torch.random.fork_rng():
        torch.manual_seed(111)
        return tiny_reasoner(config)


class LLMTrainingTests(unittest.TestCase):
    def test_compact_roundtrip_and_missing_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained", offline_backbone):
            config = config_at(Path(directory))
            model, catalog = run.build_model(config, torch.device("cpu"))
            batch, seen = collate_text_rows(load_rows(config, catalog, "train"))
            model(batch)["total"].backward()
            state = run.compact_state(model)
            self.assertTrue(any("lora_" in key for key in state))
            self.assertFalse(any("backbone" in key and "lora_" not in key for key in state))
            second, _ = run.build_model(config, torch.device("cpu"))
            run.restore_state(second, state)
            model.eval()
            second.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model.encode(batch)["query"], second.encode(batch)["query"]))
            missing = dict(state)
            missing.pop(next(k for k in state if "lora_" in k))
            with self.assertRaises(RuntimeError):
                run.restore_state(second, missing)
            unlabelled = replace(batch, positive_items=torch.full_like(batch.positive_items, 999))
            self.assertTrue(torch.equal(model.rank(batch, seen, 5), model.rank(unlabelled, seen, 5)))

    def test_short_accumulation_group_resume_and_immutable_test(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained", offline_backbone):
            config = config_at(Path(directory))
            run.train(config, torch.device("cpu"))
            resumed_config = replace(config, output_dir=config.output_dir.parent / "resume")
            partial = run.train(resumed_config, torch.device("cpu"), max_epochs_this_run=1)
            self.assertFalse(partial["complete"])
            final = run.train(resumed_config, torch.device("cpu"))
            self.assertTrue(final["complete"])
            a = torch.load(config.output_dir / "last_checkpoint.pt", weights_only=True)
            b = torch.load(resumed_config.output_dir / "last_checkpoint.pt", weights_only=True)
            self.assertEqual(a["history"][0]["optimizer_steps"], 1)
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            result = run.evaluate_checkpoint(config, torch.device("cpu"), "test")
            again = run.evaluate_checkpoint(config, torch.device("cpu"), "test")
            self.assertEqual(result, again)
            with self.assertRaises(RuntimeError):
                run.evaluate_checkpoint(replace(config, generation_score_weight=.25),
                                        torch.device("cpu"), "test")

    def test_identity_includes_smoothing_revision_and_rejects_unpinned(self):
        with tempfile.TemporaryDirectory() as directory:
            config = config_at(Path(directory))
            identity = run.signature(config)
            self.assertNotEqual(identity, run.signature(replace(config, label_smoothing=.1)))
            self.assertNotEqual(identity, run.signature(replace(config, llm_revision="b" * 40)))
            with self.assertRaises(ValueError):
                run.signature(replace(config, llm_revision="main"))


if __name__ == "__main__":
    unittest.main()
