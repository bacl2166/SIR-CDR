import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments import run_amazon_recommendation as runner
from experiments.continue_amazon_assets import ResourceUnavailable


class AmazonRecommendationRunnerTests(unittest.TestCase):
    def card(self, index=3, busy=False):
        return {"index": index, "uuid": f"GPU-{index}", "memory_free_mib": 45000,
                "utilization": 100 if busy else 0, "pids": [1] if busy else []}

    def test_disk_gate_accepts_reported_space_without_falsifying_capacity(self):
        with patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=int(29.242 * 2**30))), \
                patch.object(runner.time, "sleep") as sleep:
            free = runner.wait_for_disk(Path("/tmp"), 60)
            self.assertAlmostEqual(free, 29.242, places=3)
            sleep.assert_not_called()
        self.assertEqual(runner.MINIMUM_FREE_DISK_GIB, 24)

    def test_disk_gate_waits_below_24_and_accepts_exact_boundary(self):
        with patch.object(runner.shutil, "disk_usage", side_effect=[
                SimpleNamespace(free=24 * 2**30 - 1), SimpleNamespace(free=24 * 2**30)]), \
                patch.object(runner.time, "sleep") as sleep:
            self.assertEqual(runner.wait_for_disk(Path("/tmp"), 60), 24)
            sleep.assert_called_once_with(60)

    def test_disk_policy_has_a_separate_batch_and_preserves_other_training_settings(self):
        with tempfile.TemporaryDirectory() as d, patch.object(runner, "source", return_value={"commit": "new"}):
            root = Path(d)
            alignment = root / "alignment.json"
            alignment.write_text("{}")
            record = runner.request(root, runner.DEFAULT_BATCH_ID, alignment)
            self.assertEqual(record["minimum_free_disk_gib"], 24)
            self.assertNotEqual(runner.DEFAULT_BATCH_ID, "amazon-expansion-dual-ndcg-v1")
            cfg = runner.config_for(root, runner.DEFAULT_BATCH_ID, runner.jobs()[0])
            self.assertEqual(cfg.learning_rate, .0001)
            self.assertEqual(cfg.selection_metric, runner.policy.MEAN_METRIC)
            self.assertEqual(len(runner.jobs()), 12)

    def test_plan_has_twelve_jobs_and_dual_objective_without_changing_phones_recipe(self):
        jobs = runner.jobs()
        self.assertEqual(len(jobs), 12)
        self.assertEqual({j["seed"] for j in jobs}, {42, 43, 44})
        cfg = runner.config_for(Path("/tmp/test").resolve(), "new-batch", jobs[0])
        self.assertEqual(cfg.selection_metric, runner.policy.MEAN_METRIC)
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.gradient_accumulation_steps, 2)
        self.assertEqual(cfg.learning_rate, .0001)
        self.assertEqual(cfg.max_epochs, 30)

    def test_missing_assets_wait_and_invalid_assets_are_not_treated_as_busy(self):
        with patch.object(runner, "asset_seals", side_effect=[None, {"ready": True}]), patch.object(runner.time, "sleep") as sleep:
            self.assertEqual(runner.wait_for_assets(Path("/tmp"), Path("alignment"), 60), {"ready": True})
            sleep.assert_called_once_with(60)
        with patch.object(runner, "asset_seals", side_effect=RuntimeError("Seal changed")):
            with self.assertRaisesRegex(RuntimeError, "Seal changed"):
                runner.wait_for_assets(Path("/tmp"), Path("alignment"), 60)

    def test_final_seal_waits_for_asset_controller_to_finish_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for pair in runner.PAIRS:
                path = runner.shared.paths(root, pair)["control"] / "ready.json"
                path.parent.mkdir(parents=True)
                path.write_text("{")
            with runner.recovery.operation_lock(root / "artifacts/diagnostics/amazon-expansion-assets-v1/controller"):
                self.assertIsNone(runner.asset_seals(root, root / "alignment.json"))

    def test_restarted_controller_waits_for_owned_batch_workers_before_new_admissions(self):
        root = Path("/tmp/experiment").resolve()
        argv = ["python", "/tool/experiments/run_amazon_recommendation.py", "--worker", "validation",
                "--root", str(root), "--batch-id", "batch"]
        processes = [{"pid": 10, "argv": argv, "state": "S"},
                     {"pid": 11, "argv": argv[:-1] + ["other"], "state": "S"},
                     {"pid": 12, "argv": argv, "state": "Z"}]
        self.assertEqual([p["pid"] for p in runner.active_workers(root, "batch", processes)], [10])
        with patch.object(runner, "active_workers", side_effect=[[processes[0]], []]), \
                patch.object(runner.time, "sleep") as sleep:
            runner.wait_for_workers(root, "batch", 60)
            sleep.assert_called_once_with(60)

    def test_queue_waits_refreshes_respects_affinity_and_skips_completed(self):
        with tempfile.TemporaryDirectory() as d:
            waiting = {"busy": True}
            calls = []
            def inventory():
                return [self.card(2), self.card(3, waiting["busy"])]
            runner.run_queue([
                {"pair": "done", "seed": 42},
                {"pair": "pending", "seed": 43, "gpu_uuid": "GPU-3"},
            ], lambda j, c: calls.append((j["pair"], c["index"])), Path(d),
                max_workers=2, inventory=inventory, resources=lambda _: None, poll=.01,
                sleep=lambda _: waiting.update(busy=False),
                completed=lambda j: j["pair"] == "done")
            self.assertEqual(calls, [("pending", 3)])

    def test_prelaunch_resource_race_is_requeued(self):
        with tempfile.TemporaryDirectory() as d:
            worker = Mock(side_effect=[ResourceUnavailable("occupied"), None])
            runner.run_queue([{"pair": "pair", "seed": 42}], worker, Path(d),
                max_workers=1, inventory=lambda: [self.card()], resources=lambda _: None,
                poll=.01, sleep=Mock())
            self.assertEqual(worker.call_count, 2)

    def test_failure_during_probe_does_not_launch_sibling(self):
        with tempfile.TemporaryDirectory() as d:
            release = threading.Event()
            probes, calls = [], []
            def inventory():
                probes.append(1)
                if len(probes) == 2:
                    release.set()
                    time.sleep(.1)
                return [self.card(2), self.card(3)]
            def worker(job, card):
                calls.append(job["pair"])
                release.wait(2)
                raise RuntimeError("failed")
            with self.assertRaisesRegex(RuntimeError, "failed"):
                runner.run_queue([{"pair": "first", "seed": 42}, {"pair": "second", "seed": 42}],
                    worker, Path(d), max_workers=2, inventory=inventory,
                    resources=lambda _: None, poll=.01, sleep=Mock())
            self.assertEqual(calls, ["first"])

    def test_freeze_requires_all_twelve_and_rejects_changed_binding(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            runner.bind(folder / "declaration.json", {"test": "declaration"})
            bindings = [{**job, "files": {"checkpoint": f"hash{n}"},
                         "test_identity": {"binding": n}} for n, job in enumerate(runner.jobs())]
            with self.assertRaisesRegex(RuntimeError, "12"):
                runner.freeze_bindings(folder, bindings[:-1])
            first = runner.freeze_bindings(folder, bindings)
            self.assertEqual(len(first["jobs"]), 12)
            self.assertEqual(first, runner.freeze_bindings(folder, bindings))
            bindings[0]["files"]["checkpoint"] = "different"
            with self.assertRaisesRegex(RuntimeError, "changed"):
                runner.freeze_bindings(folder, bindings)

    def test_completed_test_is_immutable_and_summary_has_every_seed(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            runner.bind(folder / "validation_freeze.json", {"freeze": True})
            binding = {"pair": "food_to_kitchen", "seed": 42, "test_identity": {"id": 1}}
            report = folder / "test.json"
            metrics = {f"{m}@{k}": .1 for m in ("R", "N") for k in (5, 10, 20)}
            runner.bind(report, {"state": "complete", "split": "test",
                                 "recovery_identity": {"id": 1}, "metrics": metrics})
            runner.record_test(folder, binding, report)
            payload = json.loads(report.read_text())
            payload["metrics"]["N@5"] = .9
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "changed"):
                runner.record_test(folder, binding, report)
            rows = [{**job, "metrics": dict(metrics)} for job in runner.jobs()]
            result = runner.summarize_rows(rows)
            self.assertEqual(len(result), 4)
            self.assertEqual(result[0]["metrics"]["N@5"]["std"], 0)
            with self.assertRaisesRegex(RuntimeError, "seeds"):
                runner.summarize_rows(rows[:-1])

    def test_controller_runs_twelve_validations_before_test_and_restart_reuses_everything(self):
        # Exercise the real controller, queues, freeze, receipts and summary.
        # Only external GPU/model/asset execution is replaced with small fixtures.
        with tempfile.TemporaryDirectory() as d:
            root, batch = Path(d), "integration"
            alignment = root / "alignment.json"
            alignment.write_text("{}")
            args = SimpleNamespace(root=root, batch_id=batch, alignment=alignment, poll_seconds=.01)
            for pair in runner.PAIRS:
                control = runner.shared.paths(root, pair)["control"]
                control.mkdir(parents=True)
                (control / "ready.json").write_text("{}")
            folder = runner.folder_for(root, batch)
            calls = []
            queue = runner.run_queue
            def binding(root, batch, job, **kwargs):
                job = {"pair": job["pair"], "seed": job["seed"]}
                return {**job, "files": {runner.key(job): runner.sha(
                    runner.job_folder(root, batch, job) / "runtime.json")},
                    "test_identity": {"id": runner.key(job)}}
            def commands(command_list, cwd, log, gpu):
                command = command_list[0]
                phase = command[command.index("--worker") + 1]
                job = {"pair": command[command.index("--pair") + 1],
                       "seed": int(command[command.index("--seed") + 1])}
                calls.append((phase, runner.key(job)))
                run = runner.job_folder(root, batch, job)
                if phase == "smoke":
                    runner.bind(runner.smoke_path(root, batch, job["pair"]), {
                        "report": {"optimizer_steps": 0, "peak_cuda_GiB": 1},
                        "source": {"commit": "fixed"}, "ready_sha256": runner.sha(
                            runner.shared.paths(root, job["pair"])["control"] / "ready.json")})
                elif phase == "validation":
                    runner.bind(runner.affinity_path(root, batch, job), {**job, "gpu_uuid": gpu})
                    runner.bind(run / "training/manifest.json", {"complete": True})
                    runner.bind(run / "validation/verification.json", {"verified": True})
                else:
                    self.assertEqual(sum(c[0] == "validation" for c in calls), 12)
                    self.assertTrue((folder / "validation_freeze.json").is_file())
                    runner.bind(run / "training/test.json", {"state": "complete", "split": "test",
                        "recovery_identity": binding(root, batch, job)["test_identity"],
                        "metrics": {m: .1 for m in runner.METRICS}})
            def injected_queue(group, worker, status, **kwargs):
                return queue(group, worker, status, **kwargs,
                    inventory=lambda: [self.card(2), self.card(3)], resources=lambda _: None)
            with patch.object(runner, "source", return_value={"commit": "fixed"}), \
                    patch.object(runner, "wait_for_assets", return_value={}), \
                    patch.object(runner.llm, "check_cached_model"), \
                    patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=64 * 2**30)), \
                    patch.object(runner, "query_gpu_inventory", side_effect=lambda: [self.card(2), self.card(3)]), \
                    patch.object(runner, "run_queue", side_effect=injected_queue), \
                    patch.object(runner, "collect_binding", side_effect=binding), \
                    patch.object(runner, "run_commands", side_effect=commands):
                runner.run(args)
                self.assertEqual(sum(c[0] == "test" for c in calls), 12)
                self.assertEqual(len(json.loads((folder / "results.json").read_text())["per_seed"]), 12)
                count = len(calls)
                runner.run(args)
                self.assertEqual(len(calls), count)
