import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cdr_framework.text_config import TextCDRConfig
from cdr_framework.text_training import train
from experiments.sweep_text_fusion import best_result, sweep


def fixture(root: Path) -> TextCDRConfig:
    data, tokenizer = root / "data", root / "tokenizer"
    data.mkdir()
    tokenizer.mkdir()
    target = "Clothing_Shoes_and_Jewelry"
    mappings = {
        "target_domain": target,
        "id_to_item": [None]
        + [{"domain": "Sports_and_Outdoors"} for _ in range(3)]
        + [{"domain": target} for _ in range(8)],
    }
    (data / "mappings.json").write_text(json.dumps(mappings))
    rows = [
        dict(user_id=0, timestamp=5, source_items=[1, 2], target_items=[4, 5], positive_target_item=6),
        dict(user_id=1, timestamp=6, source_items=[2, 3], target_items=[5, 7], positive_target_item=8),
    ]
    for split in ("train", "validation", "test"):
        (data / f"{split}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    torch.manual_seed(31)
    latents = torch.randn(12, 4)
    latents[0].zero_()
    semantic_ids = torch.randint(0, 5, (12, 2))
    semantic_ids[0].fill_(-1)
    torch.save(latents, tokenizer / "item_latents.pt")
    torch.save(semantic_ids, tokenizer / "semantic_ids.pt")
    torch.save({"schema_version": 2, "codebooks": torch.randn(2, 5, 4)}, tokenizer / "tokenizer.pt")
    (tokenizer / "quality_report.json").write_text('{"passed": true}')
    return TextCDRConfig(
        processed_dir=data,
        tokenizer_dir=tokenizer,
        output_dir=root / "run",
        hidden_dim=4,
        max_epochs=1,
        evaluation_every=1,
        batch_size=2,
        evaluation_batch_size=2,
        rerank_candidates=8,
        decode_chunk_size=3,
        top_ks=(5, 10),
        max_sequence_length=2,
        beam_size=20,
        dropout=0.0,
    )


class TextFusionSweepTests(unittest.TestCase):
    def test_selection_uses_ndcg_and_prefers_smaller_weight_on_ties(self):
        rows = [
            {"generation_score_weight": 0.5, "NDCG@10": 0.2, "HR@10": 0.3},
            {"generation_score_weight": 1.0, "NDCG@10": 0.2, "HR@10": 0.8},
            {"generation_score_weight": 2.0, "NDCG@10": 0.1, "HR@10": 0.9},
        ]
        self.assertEqual(best_result(rows), rows[0])

    def test_sweep_preserves_checkpoint_and_resumes_completed_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            train(config, torch.device("cpu"))
            checkpoint = config.output_dir / "best_model.pt"
            original = checkpoint.read_bytes()

            result = sweep(config, torch.device("cpu"), weights=[0.0, 0.5, 1.0], batch_size=2)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["results"]), 3)
            self.assertEqual(original, checkpoint.read_bytes())

            with patch(
                "experiments.sweep_text_fusion.evaluate",
                side_effect=AssertionError("Completed weight was evaluated twice"),
            ):
                repeated = sweep(config, torch.device("cpu"), weights=[0.0, 0.5, 1.0], batch_size=2)
            self.assertEqual(result, repeated)


if __name__ == "__main__":
    unittest.main()
