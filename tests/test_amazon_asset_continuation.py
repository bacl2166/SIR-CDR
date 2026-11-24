import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

import torch

from cdr_framework.datasets.gencdr_protocol import sha
from experiments import continue_amazon_assets as continuation
from experiments import run_gencdr_shared as shared


class AssetContinuationTests(unittest.TestCase):
    def card(self, index=5, busy=False):
        return {"index": index, "uuid": f"GPU-{index}", "pids": [999] if busy else [],
                "utilization": 100 if busy else 0, "memory_free_mib": 45000}

    def test_waits_then_uses_newly_freed_gpu_and_runs_tokenizer_directly(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            state = {"free": False}
            def pause(_):
                state["free"] = True
            worker = Mock(side_effect=lambda job, card: calls.append((job["phase"], card["index"])))
            continuation.schedule_stages(
                [{"pair": "food_to_kitchen", "phase": "tokenize"}], worker,
                Path(temp), max_workers=1, poll=0.01,
                inventory=lambda: [self.card(busy=not state["free"])],
                resources=lambda reserved: None, sleep=pause,
            )
            self.assertEqual(calls, [("tokenize", 5)])

    def test_stage_transition_can_change_cards(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def inventory():
                return [self.card(4, busy=bool(calls)), self.card(5)]
            def worker(job, card):
                calls.append((job["phase"], card["index"]))
            continuation.schedule_stages(
                [{"pair": "food_to_kitchen", "phase": "embedding"}], worker,
                Path(temp), max_workers=1, poll=0.01, inventory=inventory,
                resources=lambda reserved: None, sleep=Mock(),
            )
            self.assertEqual(calls, [("embedding", 4), ("tokenize", 5)])

    def test_prelaunch_occupancy_change_requeues_without_failing_study(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def worker(job, card):
                calls.append(card["index"])
                if len(calls) == 1:
                    raise continuation.ResourceUnavailable("GPU changed occupancy")
            def inventory():
                return [self.card(4, busy=bool(calls)), self.card(5)]
            continuation.schedule_stages(
                [{"pair": "food_to_kitchen", "phase": "tokenize"}], worker,
                Path(temp), max_workers=1, poll=0.01, inventory=inventory,
                resources=lambda reserved: None, sleep=Mock(),
            )
            self.assertEqual(calls, [4, 5])
            record = json.loads((Path(temp) / "food_to_kitchen_tokenize.json").read_text())
            self.assertEqual(record["state"], "complete")

    def test_real_worker_failure_is_recorded_and_never_reported_as_gpu_wait(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "Child exit -9"):
                continuation.schedule_stages(
                    [{"pair": "food_to_kitchen", "phase": "tokenize"}],
                    Mock(side_effect=RuntimeError("Child exit -9")), folder,
                    max_workers=1, poll=0.01, inventory=lambda: [self.card()],
                    resources=lambda reserved: None, sleep=Mock(),
                )
            record = json.loads((folder / "food_to_kitchen_tokenize.json").read_text())
            self.assertEqual(record["state"], "failed")
            self.assertEqual(record["error"], "Child exit -9")

    def test_failure_during_admission_stops_new_jobs_even_with_empty_message(self):
        with tempfile.TemporaryDirectory() as temp:
            release = threading.Event()
            calls = []
            snapshots = []
            def inventory():
                snapshots.append(1)
                if len(snapshots) == 2:
                    release.set()
                    time.sleep(0.1)
                return [self.card(4), self.card(5)]
            def worker(job, card):
                calls.append(job["pair"])
                release.wait(2)
                raise RuntimeError("")
            with self.assertRaisesRegex(RuntimeError, "Asset stage failed"):
                continuation.schedule_stages(
                    [{"pair": "first", "phase": "tokenize"},
                     {"pair": "second", "phase": "tokenize"}], worker, Path(temp),
                    max_workers=2, poll=0.01, inventory=inventory,
                    resources=lambda reserved: None, sleep=Mock(),
                )
            self.assertEqual(calls, ["first"])

    def test_complete_embedding_is_checked_without_rewriting_old_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = shared.paths(root, "food_to_kitchen")
            p["processed"].mkdir(parents=True)
            p["embedding"].mkdir(parents=True)
            p["control"].mkdir(parents=True)
            (p["processed"] / "item_texts.jsonl").write_text('{"item_id":1,"text":"one","domain":"S"}\n')
            (p["processed"] / "shared_manifest.json").write_text('{}')
            model = {"revision": "a" * 40, "model_config_sha256": "b" * 64}
            (p["control"] / "embedding_model.json").write_text(json.dumps(model))
            identity = shared.asset_identity(root, "food_to_kitchen")
            manifest = {**identity, "schema_version": 1, "model": "Qwen/Qwen3-Embedding-8B",
                        "provider": "qwen3_local", "input_sha256": sha(p["processed"] / "item_texts.jsonl"),
                        "model_revision": model["revision"], "model_config_sha256": model["model_config_sha256"],
                        "dimension": 768, "normalized": True, "max_sequence_length": 512,
                        "item_count": 1, "batch_size": 2, "completed_batches": 1,
                        "matrix_shape": [2, 768], "dtype": "float32", "padding_row_is_zero": True}
            path = p["embedding"] / "manifest.json"
            path.write_text(json.dumps(manifest))
            tensor = torch.ones(2, 768)
            tensor[1] /= tensor[1].norm()
            tensor[0] = 0
            torch.save(tensor, p["embedding"] / "text_embeddings.pt")
            (p["embedding"] / "chunks").mkdir()
            chunk = p["embedding"] / "chunks/batch-000000.pt"
            torch.save({"item_ids": torch.tensor([1]), "embeddings": tensor[1:]}, chunk)
            (p["embedding"] / "item_index.json").write_text(json.dumps({
                "padding_item_id": 0, "row_is_item_id": True, "item_ids": [1]}))
            before = path.read_bytes()
            self.assertTrue(continuation.embedding_complete(root, "food_to_kitchen"))
            self.assertEqual(path.read_bytes(), before)
            tensor[0, 0] = 1
            torch.save(tensor, p["embedding"] / "text_embeddings.pt")
            with self.assertRaisesRegex(RuntimeError, "embedding matrix"):
                continuation.embedding_complete(root, "food_to_kitchen")
            tensor[0, 0] = 0
            tensor[1] = 0
            torch.save(tensor, p["embedding"] / "text_embeddings.pt")
            with self.assertRaisesRegex(RuntimeError, "normalized"):
                continuation.embedding_complete(root, "food_to_kitchen")
            tensor[1, 0] = 1
            torch.save(tensor, p["embedding"] / "text_embeddings.pt")
            with self.assertRaisesRegex(RuntimeError, "Cached embedding batch"):
                continuation.embedding_complete(root, "food_to_kitchen")
