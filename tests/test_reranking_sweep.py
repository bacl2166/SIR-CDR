import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cdr_framework.experiment_config import RecommendationTrainingConfig
from cdr_framework.formal_training import run_formal_training
from experiments.sweep_reranking import sweep, best_result


class SweepTests(unittest.TestCase):
    def test_selection_uses_ndcg_not_hr(self):
        rows = [dict(candidates=200, generation_weight=1, **{"HR@10": .5, "NDCG@10": .1}),
                dict(candidates=500, generation_weight=.5, **{"HR@10": .3, "NDCG@10": .2})]
        self.assertEqual(best_result(rows), rows[1])

    def test_sweep_reuses_checkpoint_and_completed_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, tokenizer = root / "data", root / "tokenizer"
            data.mkdir()
            tokenizer.mkdir()
            target = "Clothing_Shoes_and_Jewelry"
            mappings = {"target_domain": target, "id_to_item": [None, {"domain": "Sports_and_Outdoors"}] +
                        [{"domain": target} for _ in range(25)]}
            (data / "mappings.json").write_text(json.dumps(mappings))
            row = dict(user_id=0, timestamp=1, source_items=[1], target_items=[2], positive_target_item=3)
            for split in ("train", "validation", "test"):
                (data / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
            torch.manual_seed(9)
            latents = torch.randn(27, 4)
            latents[0].zero_()
            ids = torch.randint(0, 4, (27, 2))
            ids[0].fill_(-1)
            torch.save(latents, tokenizer / "item_latents.pt")
            torch.save(ids, tokenizer / "semantic_ids.pt")
            torch.save({"schema_version": 2, "codebooks": torch.randn(2, 4, 4)}, tokenizer / "tokenizer.pt")
            (tokenizer / "quality_report.json").write_text('{"passed": true}')
            config = RecommendationTrainingConfig(processed_dir=data, tokenizer_dir=tokenizer,
                output_dir=root / "out", hidden_dim=4, max_epochs=1, evaluation_every=1,
                batch_size=1, evaluation_batch_size=1, rerank_candidates=20)
            run_formal_training(config, device="cpu")
            original = (config.output_dir / "best_model.pt").read_bytes()
            result = sweep(config, torch.device("cpu"), [20], [0, 1], 1)
            self.assertEqual(len(result["results"]), 2)
            self.assertTrue(result["complete"])
            with patch("experiments.sweep_reranking.SIRCDRRecommender.rank_full_catalog",
                       side_effect=AssertionError("Completed group rerun")):
                repeated = sweep(config, torch.device("cpu"), [20], [0, 1], 1)
            self.assertEqual(result, repeated)
            self.assertEqual(original, (config.output_dir / "best_model.pt").read_bytes())
