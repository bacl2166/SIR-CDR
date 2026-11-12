import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from cdr_framework import llm_training
from cdr_framework.catalog_generation import CatalogTrie
from cdr_framework.text_data import TextRow
from test_catalog_generation import TableDecoder
from cdr_framework.formal_training import _sha256
from experiments import diagnose_sports_dual_recall as diagnostic
from test_llm_training import config_at, offline_backbone


class DualRecallTests(unittest.TestCase):
    def test_fixed_budget_deduplicates_and_refills_in_retrieval_order(self):
        retrieval = torch.tensor([[1, 2, 3, 4], [3, 2, 0, 0], [1, 2, 3, 4]])
        generated = torch.tensor([[2, 8, 8], [3, 0, 0], [4, 3, 0]])
        merged = diagnostic.merge_candidates(retrieval, generated)
        self.assertEqual(merged.tolist(), [[1, 2, 3, 8], [3, 2, 0, 0], [1, 2, 3, 4]])

    def test_paired_metrics_count_out_of_pool_positives_as_misses(self):
        positives = torch.tensor([2, 5, 7])
        pools = {
            "retrieval": torch.tensor([[1, 2], [3, 4], [6, 8]]),
            "dual": torch.tensor([[1, 2], [3, 5], [6, 8]]),
        }
        ranks = {
            "retrieval": torch.tensor([[1, 2], [3, 4], [6, 8]]),
            "dual": torch.tensor([[2, 1], [5, 3], [6, 8]]),
        }
        results = diagnostic.summarize(positives, pools, ranks)
        self.assertEqual(results["retrieval"]["candidate_recall"], 1 / 3)
        self.assertEqual(results["dual"]["HR@5"], 2 / 3)
        self.assertEqual(results["dual"]["NDCG@10"], 2 / 3)
        self.assertEqual(results["positive_candidate_changes"], {"added": 1, "lost": 0})

    def test_generation_rescues_missed_item_and_cannot_restore_seen_item(self):
        class ToyModel:
            target_ids = torch.tensor([1, 2, 3])
            target_index = torch.tensor([-1, 0, 1, 2, -1])
            trie = CatalogTrie(target_ids, torch.tensor([[-1], [0], [1], [2], [-1]]))
            decoder = TableDecoder({(): [0., 1., 9., 0., 0., 0.]})

            def encode(self, batch, items):
                if batch.positive_items.any():
                    raise AssertionError("Ranking must not receive labels")
                return {"prefix": torch.zeros(2, 1, 2), "query": torch.tensor([[1., 0.], [1., 0.]])}

            def sequence_scores(self, prefix, item_ids):
                return item_ids.float()

        rows = [TextRow(1, (4,), (2,), 3, frozenset({2})),
                TextRow(2, (4,), (3,), 3, frozenset({3}))]
        snapshot = {"base": torch.tensor([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., 0.]])}
        with tempfile.TemporaryDirectory() as temp, patch.multiple(
            diagnostic, CANDIDATE_BUDGET=1, GENERATION_BUDGET=1, BEAM_SIZE=1
        ):
            config = replace(config_at(Path(temp)), evaluation_batch_size=2)
            result = diagnostic.evaluate_pair(ToyModel(), rows, config, torch.device("cpu"), snapshot)
        self.assertEqual(result["retrieval"]["HR@10"], 0.0)
        self.assertEqual(result["dual"]["NDCG@5"], 0.5)
        self.assertEqual(result["dual"]["mean_candidates"], 1)
        self.assertEqual(result["positive_candidate_changes"], {"added": 1, "lost": 0})
        self.assertEqual(result["positives_in_seen"], 1)

    def test_job_uses_frozen_runtime_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            study = root / "artifacts/hparam_search/fixture"
            job = "sports_to_clothing/c06/seed42"
            runtime = study / "jobs" / job / "runtime.yaml"
            runtime.parent.mkdir(parents=True)
            config = replace(config_at(root), seed=42, output_dir=runtime.parent / "training")
            payload = {"llm_recommendation": llm_training.config_payload(config)}
            runtime.write_text(json.dumps(payload), encoding="utf-8")
            manifest = study / "study.json"
            manifest.write_text(json.dumps({"configs": {job: {
                "path": str(runtime), "sha256": _sha256(runtime), "payload": payload,
            }}}), encoding="utf-8")
            (study / "study_hash.json").write_text(json.dumps({"sha256": _sha256(manifest)}),
                                                   encoding="utf-8")
            resolved, source = diagnostic.study_job(root, "fixture", job)
            self.assertEqual(resolved, config)
            self.assertEqual(source["job"], job)
            runtime.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "runtime config differs"):
                diagnostic.study_job(root, "fixture", job)

    def test_checkpoint_run_reads_validation_and_preserves_inputs(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temp, patch(
            "cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained", offline_backbone
        ):
            root = Path(temp)
            config = replace(config_at(root), generation_score_weight=0.5)
            llm_training.train(config, device)
            ckpt = config.output_dir / "best_model.pt"
            manifest = config.output_dir / "manifest.json"
            before = [_sha256(path) for path in (ckpt, manifest)]
            output = root / "diagnostic"
            report = diagnostic.run_diagnostic(config, output, device, {"job": "fixture"})
            self.assertEqual(report["split"], "validation")
            self.assertEqual(report["results"]["retrieval"]["examples"], 2)
            self.assertEqual(report["results"]["retrieval"]["mean_candidates"],
                             report["results"]["dual"]["mean_candidates"])
            self.assertEqual(before, [_sha256(path) for path in (ckpt, manifest)])
            self.assertEqual(report, json.loads((output / "report.json").read_text(encoding="utf-8")))
            self.assertFalse((config.output_dir / "test.json").exists())
            with self.assertRaises(FileExistsError):
                diagnostic.run_diagnostic(config, output, device, {"job": "fixture"})
