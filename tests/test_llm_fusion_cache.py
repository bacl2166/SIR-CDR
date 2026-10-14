import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import torch
from unittest.mock import patch

from test_text_pipeline import fixture
from cdr_framework.text_training import build_model, load_rows
from cdr_framework.text_data import collate_text_rows
from experiments.tune_llm_fusion import collect_scores, cached_rank, score_metrics, sweep_settings


class CacheTests(unittest.TestCase):
    def test_cached_ranking_matches_direct_and_encodes_only_once(self):
        with tempfile.TemporaryDirectory() as d:
            config = replace(fixture(Path(d)), top_ks=(2, 3), selection_metric="NDCG@3", rerank_candidates=4)
            model, cat = build_model(config, torch.device("cpu"))
            rows = load_rows(config, cat, "validation")
            cache = collect_scores(model, rows, config, torch.device("cpu"), 8)
            batch, seen = collate_text_rows(rows)
            for weight in (0., .25, .5, 1.):
                model.config = replace(config, generation_score_weight=weight, fusion_normalization="zscore")
                expected = model.rank(batch, seen, 3, "hybrid")
                actual = cached_rank(cache, 4, weight, 3)
                self.assertTrue(torch.equal(expected.cpu(), actual))
                self.assertEqual(score_metrics(cache, actual, (2,3))["examples"], 2)
            self.assertEqual(len(sweep_settings()), 21)
            self.assertNotIn("test", cache)

    def test_insufficient_cache_is_rejected(self):
        cache = {"candidates": torch.tensor([[1,2]]), "retrieval": torch.tensor([[2.,1.]]),
                 "generation": torch.tensor([[1.,2.]])}
        with self.assertRaises(ValueError):
            cached_rank(cache, 3, .5, 2)

    def test_partial_cache_matches_uninterrupted_collection(self):
        with tempfile.TemporaryDirectory() as d:
            config = fixture(Path(d))
            model, cat = build_model(config,torch.device("cpu"))
            rows = load_rows(config,cat,"validation")
            full = collect_scores(model,rows,config,torch.device("cpu"),8)
            partial = collect_scores(model,rows[:1],config,torch.device("cpu"),8)
            resumed = collect_scores(model,rows,config,torch.device("cpu"),8,partial)
            for key in full:
                torch.testing.assert_close(full[key],resumed[key])

    def test_compact_checkpoint_search_verify_reuse_and_test_lock(self):
        import json
        from test_llm_training import config_at, offline_backbone
        from cdr_framework import llm_training
        from experiments.tune_llm_fusion import run
        with tempfile.TemporaryDirectory() as d, patch(
                "cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained",offline_backbone):
            config = config_at(Path(d))
            llm_training.train(config,torch.device("cpu"))
            out = Path(d)/"tuning"
            run(config,torch.device("cpu"),out,"all")
            self.assertEqual(len(json.loads((out/"search.json").read_text())["results"]),21)
            run(config,torch.device("cpu"),out,"verify")
            with patch("cdr_framework.llm_training.build_model",side_effect=AssertionError("must reuse cache")):
                run(config,torch.device("cpu"),out,"all")
            (config.output_dir/"test.json").write_text('{}')
            with self.assertRaisesRegex(RuntimeError,"already evaluated"):
                run(config,torch.device("cpu"),out,"all")
