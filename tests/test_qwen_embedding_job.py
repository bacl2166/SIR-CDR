import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from cdr_framework.embeddings.qwen import ItemText, load_item_texts, run_embedding_job
from cdr_framework.experiment_config import QwenEmbeddingConfig
from experiments.embed_items import run


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def encode_text(self, records):
        self.calls += 1
        return torch.tensor(
            [[float(len(text)), float(index + 1)] for index, text in enumerate(records)],
            dtype=torch.float32,
        )


class QwenEmbeddingJobTests(unittest.TestCase):
    def test_embedding_cli_accepts_device_override(self):
        from experiments.embed_items import build_parser

        args = build_parser().parse_args(["--config", "experiment.yaml", "--device", "cpu"])
        self.assertEqual(args.device, "cpu")

    @patch("experiments.embed_items.Qwen3EmbeddingProvider")
    def test_smoke_test_selects_local_qwen3_8b_provider(self, provider_class):
        provider_class.return_value.encode_text.return_value = torch.ones((1, 768))
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "items.jsonl"
            input_path.write_text(
                '{"item_id":1,"domain":"source","text":"running jacket"}\n',
                encoding="utf-8",
            )
            config = QwenEmbeddingConfig(
                input_path=input_path,
                output_dir=Path(directory) / "output",
                provider="qwen3_local",
                model="Qwen/Qwen3-Embedding-8B",
                dimension=768,
                batch_size=1,
                device="cuda",
                max_sequence_length=8192,
                model_path_env="QWEN3_EMBEDDING_MODEL_PATH",
            )

            self.assertEqual(run(config, smoke_test=True, force=False), 0)

        provider_class.assert_called_once_with(
            model="Qwen/Qwen3-Embedding-8B",
            dim=768,
            batch_size=1,
            device="cuda",
            max_sequence_length=8192,
            model_path_env="QWEN3_EMBEDDING_MODEL_PATH",
        )
    def test_loads_sorted_item_texts_without_using_ids_as_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            path.write_text(
                '{"item_id":2,"domain":"target","text":"coat"}\n'
                '{"item_id":1,"domain":"source","text":"ball"}\n',
                encoding="utf-8",
            )

            items = load_item_texts(path)

            self.assertEqual([item.item_id for item in items], [1, 2])
            self.assertEqual([item.text for item in items], ["ball", "coat"])

    def test_job_writes_padding_matrix_and_resumes_completed_batches(self):
        items = [
            ItemText(1, "source", "one"),
            ItemText(2, "source", "two"),
            ItemText(3, "target", "three"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embeddings"
            first_provider = FakeProvider()
            first = run_embedding_job(
                items,
                first_provider,
                output,
                input_sha256="abc",
                model="text-embedding-v4",
                dimension=2,
                batch_size=2,
                max_retries=0,
                retry_initial_seconds=0,
            )

            matrix = torch.load(first.embeddings_path, map_location="cpu", weights_only=True)
            self.assertEqual(tuple(matrix.shape), (4, 2))
            self.assertTrue(torch.equal(matrix[0], torch.zeros(2)))
            self.assertEqual(first_provider.calls, 2)

            second_provider = FakeProvider()
            second = run_embedding_job(
                items,
                second_provider,
                output,
                input_sha256="abc",
                model="text-embedding-v4",
                dimension=2,
                batch_size=2,
                max_retries=0,
                retry_initial_seconds=0,
            )

            self.assertEqual(second_provider.calls, 0)
            self.assertEqual(second.resumed_batches, 2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["matrix_shape"], [4, 2])

    def test_resume_rejects_changed_local_model_identity(self):
        items = [ItemText(1, "source", "one")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embeddings"
            run_embedding_job(
                items,
                FakeProvider(),
                output,
                input_sha256="abc",
                model="Qwen/Qwen3-Embedding-8B",
                dimension=2,
                batch_size=1,
                max_retries=0,
                retry_initial_seconds=0,
                identity_metadata={"provider": "qwen3_local", "max_sequence_length": 8192},
            )

            with self.assertRaisesRegex(RuntimeError, "max_sequence_length"):
                run_embedding_job(
                    items,
                    FakeProvider(),
                    output,
                    input_sha256="abc",
                    model="Qwen/Qwen3-Embedding-8B",
                    dimension=2,
                    batch_size=1,
                    max_retries=0,
                    retry_initial_seconds=0,
                    identity_metadata={"provider": "qwen3_local", "max_sequence_length": 4096},
                )


if __name__ == "__main__":
    unittest.main()
