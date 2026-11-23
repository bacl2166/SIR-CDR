from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cdr_framework.experiment_config import TokenizerTrainingConfig
from cdr_framework.datasets.gencdr_protocol import sha
from experiments import run_gencdr_shared as shared
from experiments.prepare_amazon_assets import reference_recipe, bind_embedding, pair_registration, bind, stage
from experiments import prepare_amazon_assets as tool


class AmazonAssetTests(unittest.TestCase):
    def fixture(self, root):
        folder = shared.paths(root, "phones_to_electronics")["control"]
        folder.mkdir(parents=True)
        model = {"identity": {}, "revision": "a" * 40, "model_config_sha256": "b" * 64}
        config = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(TokenizerTrainingConfig()).items()}
        (folder / "embedding_model.json").write_text(json.dumps(model), encoding="utf-8")
        (folder / "tokenizer_execution.json").write_text(json.dumps({"config": config}), encoding="utf-8")
        return folder

    def test_reference_tokenizer_recipe_must_match_existing_implementation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(shared, "verify_ready"):
            root = Path(temp)
            folder = self.fixture(root)
            recipe = reference_recipe(root)
            self.assertEqual(recipe["embedding"]["revision"], "a" * 40)
            file = folder / "tokenizer_execution.json"
            payload = json.loads(file.read_text(encoding="utf-8"))
            payload["config"]["max_epochs"] = 99
            file.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "tokenizer recipe"):
                reference_recipe(root)

    def test_new_receipt_uses_reference_revision_and_new_dataset_identity(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(shared, "verify_ready"):
            root = Path(temp)
            self.fixture(root)
            recipe = reference_recipe(root)
            pair = "food_to_kitchen"
            processed = shared.paths(root, pair)["processed"]
            processed.mkdir(parents=True)
            (processed / "shared_manifest.json").write_text("{}", encoding="utf-8")
            bind_embedding(root, pair, recipe)
            file = shared.paths(root, pair)["control"] / "embedding_model.json"
            current = json.loads(file.read_text(encoding="utf-8"))
            self.assertEqual(current["revision"], "a" * 40)
            self.assertEqual(current["identity"]["protocol_sha256"], sha(processed / "shared_manifest.json"))
            bind_embedding(root, pair, recipe)
            current["revision"] = "c" * 40
            file.write_text(json.dumps(current), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Embedding receipt"):
                bind_embedding(root, pair, recipe)

    def test_registration_is_scoped_and_leaves_original_pair_map_intact(self):
        before = dict(shared.PAIRS)
        with pair_registration():
            self.assertEqual(shared.PAIRS["food_to_kitchen"],
                             ("Grocery_and_Gourmet_Food", "Home_and_Kitchen"))
            self.assertEqual(shared.PAIRS["electronics_to_beauty"], ("Electronics", "Beauty"))
        self.assertEqual(shared.PAIRS, before)

    def test_execution_binding_roundtrips_and_rejects_recipe_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recipe.json"
            payload = {"domains": ["Grocery_and_Gourmet_Food", "Home_and_Kitchen"]}
            bind(path, payload)
            before = path.read_bytes()
            bind(path, payload)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(RuntimeError, "execution binding"):
                bind(path, {"domains": list(reversed(payload["domains"]))})

    def test_child_verifies_gpu_before_loading_and_dispatches_original_stage(self):
        import experiments.prepare_amazon_assets as tool
        with tempfile.TemporaryDirectory() as temp, patch.object(shared, "verify_gpu_binding") as gpu, \
                patch.object(tool, "reference_recipe", return_value={}), \
                patch.object(tool, "verify_pair_data"), patch.object(tool, "load_alignment"), \
                patch.object(tool, "read"), patch.object(tool, "bind_embedding"), \
                patch.object(shared, "embedding") as embedding, patch.object(shared, "tokenize") as tokenize:
            root = Path(temp)
            stage(root, "food_to_kitchen", "embedding", Path("alignment.json"))
            gpu.assert_called_once()
            embedding.assert_called_once_with(root, "food_to_kitchen", "cuda:0")
            tokenize.assert_not_called()
            stage(root, "electronics_to_beauty", "tokenize", Path("alignment.json"))
            tokenize.assert_called_once_with(root, "electronics_to_beauty", "cuda:0")

    def test_ready_reuse_rejects_different_tokenizer_training_parameters(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(shared, "verify_ready"):
            root = Path(temp)
            self.fixture(root)
            recipe = reference_recipe(root)
            folder = shared.paths(root, "food_to_kitchen")["control"]
            folder.mkdir(parents=True)
            payload = {"config": {**asdict(TokenizerTrainingConfig()), "max_epochs": 1}}
            payload["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in payload["config"].items()}
            (folder / "tokenizer_execution.json").write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(tool, "bind_embedding"):
                with self.assertRaisesRegex(RuntimeError, "tokenizer recipe"):
                    tool.verify_pair_recipe(root, "food_to_kitchen", recipe)

    def test_swapped_direction_cannot_consume_alignment_report(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(tool, "verify_shared"):
            root = Path(temp)
            pair = "food_to_kitchen"
            folder = shared.paths(root, pair)["processed"]
            folder.mkdir(parents=True)
            (folder / "shared_manifest.json").write_text("{}", encoding="utf-8")
            swapped = {"identity": {"source_domain": "Home_and_Kitchen",
                                    "target_domain": "Grocery_and_Gourmet_Food"}}
            (folder / "protocol.json").write_text(json.dumps(swapped), encoding="utf-8")
            report = {"directions": {pair: {"shared_manifest_sha256": sha(folder / "shared_manifest.json")}}}
            with self.assertRaisesRegex(RuntimeError, "domain roles"):
                tool.verify_pair_data(root, pair, report, {})

    def test_eight_idle_cards_fit_original_scheduler_limit_and_keep_affinity(self):
        cards = [{"index": i, "uuid": f"gpu{i}", "pids": [], "utilization": 0,
                  "memory_free_mib": 45000} for i in range(8)]
        selected = tool.select_cards(cards, [{"gpu_uuid": "gpu7"}])
        from cdr_framework.experiment_control import parse_gpu_ids
        self.assertEqual(len(parse_gpu_ids([c["index"] for c in selected])), 6)
        self.assertIn("gpu7", [c["uuid"] for c in selected])

    def test_gpu_taken_between_stages_prevents_tokenizer_launch(self):
        idle = {"index": 6, "uuid": "GPU-six", "pids": [], "utilization": 0,
                "memory_free_mib": 45000}
        busy = {**idle, "pids": [999]}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(tool, "query_gpu_inventory", side_effect=[[idle], [busy]]), \
                patch.object(tool, "run_commands", return_value={}) as commands:
            with self.assertRaisesRegex(RuntimeError, "no longer available"):
                tool.run_stages(Path(temp), "food_to_kitchen", 6, Path("alignment.json"))
            self.assertEqual(commands.call_count, 1)
            command = commands.call_args.args[0][0]
            self.assertEqual(command[command.index("--stage") + 1], "embedding")
