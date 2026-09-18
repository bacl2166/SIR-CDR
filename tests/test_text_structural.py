"""Tests for the dual structural injection wiring in the text-only SIR-CDR v3 model."""

import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

import torch

from cdr_framework.text_config import TextCDRConfig
from cdr_framework.text_data import collate_text_rows, load_text_rows
from cdr_framework.text_training import build_model, variant_config, VARIANTS


def fixture(root):
    data, tokenizer = root / "data", root / "tokenizer"
    data.mkdir()
    tokenizer.mkdir()
    target = "Clothing_Shoes_and_Jewelry"
    mappings = {"target_domain": target, "id_to_item": [None] +
        [{"domain": "Sports_and_Outdoors"} for _ in range(3)] + [{"domain": target} for _ in range(8)]}
    (data / "mappings.json").write_text(json.dumps(mappings))
    rows = [dict(user_id=0, timestamp=5, source_items=[1, 2], target_items=[4, 5, 6], positive_target_item=7),
            dict(user_id=1, timestamp=6, source_items=[2, 3], target_items=[5, 8], positive_target_item=9)]
    for split in ("train", "validation", "test"):
        (data / f"{split}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    torch.manual_seed(23)
    latents = torch.randn(12, 4)
    latents[0] = 0
    ids = torch.randint(0, 5, (12, 2))
    ids[0] = -1
    torch.save(latents, tokenizer / "item_latents.pt")
    torch.save(ids, tokenizer / "semantic_ids.pt")
    torch.save({"schema_version": 2, "codebooks": torch.randn(2, 5, 4)}, tokenizer / "tokenizer.pt")
    (tokenizer / "quality_report.json").write_text('{"passed": true}')
    return TextCDRConfig(processed_dir=data, tokenizer_dir=tokenizer, output_dir=root / "run",
        hidden_dim=4, max_epochs=2, evaluation_every=1, batch_size=2, evaluation_batch_size=2,
        rerank_candidates=8, decode_chunk_size=3, top_ks=(5, 10), max_sequence_length=2,
        beam_size=50, dropout=0.1)


class TextStructuralInjectionTests(unittest.TestCase):
    def test_new_architecture_forward_backward_activates_all_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            model, catalog = build_model(config, torch.device("cpu"))
            rows = load_text_rows(config.processed_dir / "train.jsonl", config.max_sequence_length,
                                  len(catalog.item_latents), catalog.target_item_ids)
            batch, _ = collate_text_rows(rows)
            losses = model(batch)
            self.assertIn("lsep", losses)
            self.assertIn("proto_orth", losses)
            self.assertTrue(torch.isfinite(losses["total"]))
            losses["total"].backward()
            self.assertGreater(losses["lsep"].item(), 0)
            self.assertGreater(losses["proto_orth"].item(), 0)
            for module in (model.prototype, model.cd_injector, model.sp_injector, model.codebook_summary):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()),
                                type(module).__name__)

    def test_item_states_provides_prototype_tokens_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            model, _ = build_model(config, torch.device("cpu"))
            states = model.item_states()
            self.assertEqual(set(states), {"base", "shared", "source", "target", "proto"})
            proto = states["proto"]
            for key in ("proto_shared", "proto_source", "proto_target",
                        "tokens_shared", "tokens_source", "tokens_target"):
                self.assertIn(key, proto)
                self.assertEqual(proto[key].shape[0], len(model.content))
            no_proto = replace(config, prototype_enabled=False)
            model_no, _ = build_model(no_proto, torch.device("cpu"))
            self.assertIsNone(model_no.item_states()["proto"])

    def test_switches_zero_the_corresponding_losses(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            base_model, _ = build_model(config, torch.device("cpu"))
            batch, _ = collate_text_rows(load_text_rows(
                config.processed_dir / "train.jsonl", config.max_sequence_length,
                len(base_model.content), base_model.target_ids))
            full = base_model(batch)
            no_proto = variant_config(config, "no_proto", 42)
            model_no_proto, _ = build_model(no_proto, torch.device("cpu"))
            self.assertEqual(model_no_proto(batch)["proto_orth"].item(), 0.0)
            no_lsep = variant_config(config, "no_lsep", 42)
            model_no_lsep, _ = build_model(no_lsep, torch.device("cpu"))
            self.assertEqual(model_no_lsep(batch)["lsep"].item(), 0.0)
            self.assertNotEqual(full["proto_orth"].item(), 0.0)

    def test_new_and_legacy_variants_execute_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            for name in ("no_proto", "no_cd_inj", "no_sp_inj", "no_lsep", "no_lsh", "no_csum", "v2_legacy"):
                current = variant_config(config, name, 42)
                model, catalog = build_model(current, torch.device("cpu"))
                batch, _ = collate_text_rows(load_text_rows(
                    current.processed_dir / "train.jsonl", current.max_sequence_length,
                    len(catalog.item_latents), catalog.target_item_ids))
                loss = model(batch)["total"]
                self.assertTrue(torch.isfinite(loss), name)
                loss.backward()

    def test_all_ablation_variants_still_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            for name in VARIANTS:
                current = variant_config(config, name, 42)
                model, catalog = build_model(current, torch.device("cpu"))
                batch, _ = collate_text_rows(load_text_rows(
                    current.processed_dir / "train.jsonl", current.max_sequence_length,
                    len(catalog.item_latents), catalog.target_item_ids))
                loss = model(batch)["total"]
                self.assertTrue(torch.isfinite(loss), name)
                loss.backward()

    def test_codebook_summary_and_injectors_change_the_prefix_path(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            model, catalog = build_model(config, torch.device("cpu"))
            batch, _ = collate_text_rows(load_text_rows(
                config.processed_dir / "train.jsonl", config.max_sequence_length,
                len(catalog.item_latents), catalog.target_item_ids))
            state = model.encode(batch)
            self.assertEqual(state["prefix"].shape, (2, config.prefix_length, config.hidden_dim))
            summary = model.codebook_summary(state["shared"] + state["private"], model.centroids)
            self.assertEqual(summary.shape, (2, config.hidden_dim))
            self.assertTrue(torch.isfinite(summary).all())
            model.eval()
            for mode in ("retrieval", "hybrid", "exhaustive", "generate"):
                ranked = model.rank(batch, [{4, 5, 6, 7}, {5, 8, 9}], 10, mode=mode)
                self.assertEqual(ranked.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
