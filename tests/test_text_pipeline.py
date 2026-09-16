import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

import torch

from cdr_framework.text_config import TextCDRConfig
from cdr_framework.text_data import collate_text_rows
from cdr_framework.text_training import train, build_model, load_rows, evaluate_checkpoint, variant_config, VARIANTS


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


class TextPipelineTests(unittest.TestCase):
    def test_feedback_graph_gradients_and_label_free_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            model, catalog = build_model(config, torch.device("cpu"))
            rows = load_rows(config, catalog, "train")
            batch, seen = collate_text_rows(rows)
            self.assertNotIn(4, rows[0].target_items)
            self.assertIn(4, seen[0])
            losses = model(batch)
            losses["total"].backward()
            self.assertGreater(model.feedback_proj.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.reasoner.write_gate.weight.grad.abs().sum().item(), 0)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.graph_encoder.parameters()))
            model.eval()
            unlabelled = replace(batch, positive_items=torch.full_like(batch.positive_items, 999))
            for mode in ("retrieval", "hybrid", "exhaustive", "generate"):
                ranked = model.rank(unlabelled, seen, 10, mode=mode)
                self.assertEqual(ranked.shape, (2, 10))
                for history, hits in zip(seen, ranked.tolist()):
                    valid = [x for x in hits if x]
                    self.assertFalse(history.intersection(valid))
                    self.assertEqual(len(valid), len(set(valid)))
                    self.assertTrue(set(valid).issubset(set(catalog.target_item_ids.tolist())))
            # Identical pool covering all targets must equal exhaustive scoring.
            self.assertTrue(torch.equal(model.rank(batch, seen, 5, "hybrid"), model.rank(batch, seen, 5, "exhaustive")))

    def test_checkpoint_resume_is_identical_with_dropout(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            full = train(config, torch.device("cpu"))
            resumed_config = replace(config, output_dir=config.output_dir.parent / "resumed")
            paused = train(resumed_config, torch.device("cpu"), max_epochs_this_run=1)
            self.assertFalse(paused["complete"])
            resumed = train(resumed_config, torch.device("cpu"))
            self.assertTrue(resumed["complete"])
            a = torch.load(config.output_dir / "last_checkpoint.pt", weights_only=True)
            b = torch.load(resumed_config.output_dir / "last_checkpoint.pt", weights_only=True)
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            result = evaluate_checkpoint(config, torch.device("cpu"), "validation", "generate")
            self.assertIn("R@5", result["metrics"])
            self.assertTrue(full["complete"])

    def test_ablation_variants_execute_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            for name in VARIANTS:
                current = variant_config(config, name, 42)
                model, catalog = build_model(current, torch.device("cpu"))
                batch, _ = collate_text_rows(load_rows(current, catalog, "train"))
                loss = model(batch)["total"]
                self.assertTrue(torch.isfinite(loss), name)
                loss.backward()
