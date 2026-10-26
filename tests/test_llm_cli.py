import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import yaml

from experiments.run_llm_cdr import prepare_model, check_cached_model, absolute_config, smoke
from cdr_framework.llm_config import LLMCDRConfig


class LLMCLITests(unittest.TestCase):
    def test_cuda_smoke_initializes_selected_device_before_resetting_peak_stats(self):
        device = torch.device("cuda:0")
        initialized = []

        def select_device(selected):
            initialized.append(selected)

        def reset_peak(selected):
            if selected not in initialized:
                raise RuntimeError("Invalid device argument")

        with patch("torch.cuda.set_device", side_effect=select_device) as select, \
                patch("torch.cuda.reset_peak_memory_stats", side_effect=reset_peak) as reset, \
                patch("experiments.run_llm_cdr.llm_training.build_model",
                      side_effect=RuntimeError("model construction reached")) as build:
            with self.assertRaisesRegex(RuntimeError, "model construction reached"):
                smoke(LLMCDRConfig(), device)
            select.assert_called_once_with(device)
            reset.assert_called_once_with(device)
            build.assert_called_once()

    def test_cuda_smoke_without_index_keeps_the_current_device(self):
        with patch("torch.cuda.current_device", return_value=2) as current, \
                patch("torch.cuda.set_device") as select, \
                patch("torch.cuda.reset_peak_memory_stats") as reset, \
                patch("experiments.run_llm_cdr.llm_training.build_model",
                      side_effect=RuntimeError("model construction reached")):
            with self.assertRaisesRegex(RuntimeError, "model construction reached"):
                smoke(LLMCDRConfig(), torch.device("cuda"))
            current.assert_called_once_with()
            select.assert_called_once_with(2)
            reset.assert_called_once_with(torch.device("cuda"))

    def test_cpu_smoke_does_not_initialize_cuda_or_reset_gpu_stats(self):
        with patch("torch.cuda.set_device") as select, \
                patch("torch.cuda.reset_peak_memory_stats") as reset, \
                patch("experiments.run_llm_cdr.llm_training.build_model",
                      side_effect=RuntimeError("model construction reached")):
            with self.assertRaisesRegex(RuntimeError, "model construction reached"):
                smoke(LLMCDRConfig(), torch.device("cpu"))
            select.assert_not_called()
            reset.assert_not_called()

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
