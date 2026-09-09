import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.diagnose_recommender import training_popularity, summarize
from cdr_framework.formal_recommendation import FixedCatalog, FormalBatch, SIRCDRRecommender


class DiagnosticTests(unittest.TestCase):
    def test_popularity_does_not_count_sliding_prefix_twice(self):
        rows = [
            dict(user_id=1, timestamp=1, target_items=[3], positive_target_item=4),
            dict(user_id=1, timestamp=2, target_items=[3, 4], positive_target_item=5),
            dict(user_id=2, timestamp=1, target_items=[3], positive_target_item=4),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            self.assertEqual(training_popularity(path), {3: 2, 4: 2, 5: 1})

    def test_retrieval_skips_decoder_and_keeps_full_catalog_candidates(self):
        torch.manual_seed(12)
        latents = torch.randn(7, 4)
        latents[0].zero_()
        tokens = torch.randint(0, 5, (7, 2))
        tokens[0].fill_(-1)
        model = SIRCDRRecommender(FixedCatalog(
            latents, tokens, torch.randn(2, 5, 4), torch.tensor([3, 4, 5, 6]),
        ), hidden_dim=4).eval()
        batch = FormalBatch(torch.tensor([[1, 2]]), torch.tensor([2]),
                            torch.tensor([[3]]), torch.tensor([1]), torch.tensor([999]))
        with patch.object(model.decoder, "forward", side_effect=AssertionError("Decoder called")):
            one = model.rank_full_catalog(batch, top_k=3, candidate_chunk_size=1, generation_weight=0)
            all_at_once = model.rank_full_catalog(batch, top_k=3, candidate_chunk_size=10, generation_weight=0)
        self.assertEqual(one.tolist(), all_at_once.tolist())
        self.assertEqual(set(one[0].tolist()), {4, 5, 6})

    def test_metrics_include_missed_candidates(self):
        result = summarize([[3, 4], [4, 3]], [3, 5], (1, 2))
        self.assertEqual(result["HR@2"], 0.5)
        self.assertEqual(result["NDCG@1"], 0.5)
