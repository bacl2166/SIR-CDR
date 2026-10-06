import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml

from experiments.run_llm_cdr import prepare_model, check_cached_model, absolute_config
from cdr_framework.llm_config import LLMCDRConfig


class LLMCLITests(unittest.TestCase):
    def test_prepare_pins_revision_and_wont_overwrite_different_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / ("a" * 40)
            snapshot.mkdir()
            (snapshot / "config.json").write_text('{"model_type":"qwen3"}')
            (snapshot / "model.safetensors").write_bytes(b"fixture")
            output = root / "resolved.yaml"
            with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)):
                prepare_model(LLMCDRConfig(), output)
                config = LLMCDRConfig.from_yaml(output)
                self.assertEqual(config.llm_revision, "a" * 40)
                prepare_model(LLMCDRConfig(), output)
                with self.assertRaises(RuntimeError):
                    prepare_model(LLMCDRConfig(seed=43), output)

    def test_preflight_rejects_incomplete_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ("b" * 40)
            root.mkdir()
            (root / "config.json").write_text('{"model_type":"qwen3"}')
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"weight": "model-00001-of-00001.safetensors"}}))
            config = LLMCDRConfig(llm_revision="b" * 40)
            with patch("huggingface_hub.snapshot_download", return_value=str(root)):
                with self.assertRaises(FileNotFoundError):
                    check_cached_model(config)
                (root / "model-00001-of-00001.safetensors").write_bytes(b"fixture")
                self.assertEqual(check_cached_model(config), root)

    def test_config_paths_are_repository_relative_not_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump({"llm_recommendation": {
                "processed_dir": "data/processed/a", "tokenizer_dir": "artifacts/tokenizer/a",
                "output_dir": "artifacts/llm_cdr_v1/a"}}))
            config = absolute_config(path, root)
            self.assertEqual(config.processed_dir, root / "data/processed/a")


if __name__ == "__main__":
    unittest.main()
