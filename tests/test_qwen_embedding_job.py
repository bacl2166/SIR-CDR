import json
import tempfile
import unittest
from pathlib import Path

import torch

from cdr_framework.embeddings.qwen import ItemText, load_item_texts, run_embedding_job


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


if __name__ == "__main__":
    unittest.main()
