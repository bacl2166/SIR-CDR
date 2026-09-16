import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import torch

from cdr_framework.config import EmbeddingConfig
from cdr_framework.embeddings.api import DeepSeekTextEmbeddingProvider, QwenTextEmbeddingProvider, build_text_embedding_provider
from cdr_framework.embeddings.qwen3_local import Qwen3EmbeddingProvider
from cdr_framework.embeddings.cache import CachedEmbeddingStore
from cdr_framework.embeddings.local import DeterministicHashTextEmbeddingProvider
from cdr_framework.embeddings.providers import BaseLLMSemanticProvider


class EmbeddingProviderTest(unittest.TestCase):
    def test_qwen3_8b_provider_uses_local_override_and_normalized_mrl_dimension(self):
        calls = []

        class FakeModel:
            max_seq_length = None
            tokenizer = SimpleNamespace(padding_side="right")

            def encode(self, texts, **kwargs):
                calls.append((texts, kwargs, self.max_seq_length))
                return torch.arange(1, len(texts) * 8 + 1, dtype=torch.float32).reshape(len(texts), 8)

        def loader(model_name_or_path, **kwargs):
            calls.append((model_name_or_path, kwargs))
            return FakeModel()

        with patch.dict("os.environ", {"QWEN3_EMBEDDING_MODEL_PATH": "/models/qwen3-8b"}):
            provider = Qwen3EmbeddingProvider(
                model="Qwen/Qwen3-Embedding-8B",
                dim=3,
                device="cuda",
                max_sequence_length=4096,
                model_loader=loader,
            )
            vectors = provider.encode_text(["sports jacket", "running shoes"])

        self.assertEqual(calls[0][0], "/models/qwen3-8b")
        self.assertEqual(calls[0][1]["device"], "cuda")
        self.assertEqual(calls[0][1]["model_kwargs"]["torch_dtype"], torch.bfloat16)
        self.assertEqual(calls[0][1]["model_kwargs"]["attn_implementation"], "sdpa")
        self.assertNotIn("tokenizer_kwargs", calls[0][1])
        self.assertEqual(provider._encoder.tokenizer.padding_side, "left")
        self.assertEqual(calls[1][0], ["sports jacket", "running shoes"])
        self.assertEqual(calls[1][2], 4096)
        self.assertEqual(tuple(vectors.shape), (2, 3))
        self.assertEqual(vectors.dtype, torch.float32)
        self.assertTrue(torch.allclose(vectors.norm(dim=1), torch.ones(2), atol=1e-6))

    def test_qwen3_8b_provider_loads_model_only_when_encoding(self):
        loaded = []

        class FakeModel:
            max_seq_length = 32

            def encode(self, texts, **kwargs):
                return torch.ones((len(texts), 4))

        provider = Qwen3EmbeddingProvider(
            dim=4,
            device="cpu",
            model_loader=lambda *args, **kwargs: loaded.append((args, kwargs)) or FakeModel(),
        )
        self.assertEqual(loaded, [])
        self.assertEqual(tuple(provider.encode_text([]).shape), (0, 4))
        self.assertEqual(loaded, [])
        provider.encode_text(["coat"])
        self.assertEqual(len(loaded), 1)

    def test_cached_embedding_store_round_trips_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CachedEmbeddingStore(Path(tmp))
            tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

            self.assertFalse(store.exists("text", "amazon-sports"))
            store.save("text", "amazon-sports", tensor)

            self.assertTrue(store.exists("text", "amazon-sports"))
            self.assertTrue(torch.equal(store.load("text", "amazon-sports"), tensor))

    def test_local_text_provider_is_deterministic_and_normalized(self):
        provider = DeterministicHashTextEmbeddingProvider(dim=8)

        first = provider.encode_text(["Trail jacket", "Movie book"])
        second = provider.encode_text(["Trail jacket", "Movie book"])

        self.assertEqual(first.shape, (2, 8))
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.allclose(first.norm(dim=-1), torch.ones(2), atol=1e-6))

    def test_llm_semantic_provider_is_reserved_interface(self):
        self.assertTrue(hasattr(BaseLLMSemanticProvider, "score_cross_domain_pairs"))
        self.assertTrue(hasattr(BaseLLMSemanticProvider, "summarize_item_metadata"))

    def test_qwen_provider_calls_compatible_embedding_api(self):
        calls = []

        class Embeddings:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    data=[SimpleNamespace(index=0, embedding=[0.1, 0.2, 0.3])]
                )

        client = SimpleNamespace(embeddings=Embeddings())
        qwen = QwenTextEmbeddingProvider(
            model="text-embedding-v4", dim=3, client=client
        )

        vectors = qwen.encode_text(["hello"])

        self.assertEqual(tuple(vectors.shape), (1, 3))
        self.assertEqual(vectors.dtype, torch.float32)
        self.assertEqual(calls[0]["model"], "text-embedding-v4")
        self.assertEqual(calls[0]["dimensions"], 3)
        self.assertEqual(calls[0]["encoding_format"], "float")

    def test_deepseek_provider_remains_reserved(self):
        deepseek = DeepSeekTextEmbeddingProvider(api_key_env="DEEPSEEK_API_KEY", model="deepseek-embedding")

        with self.assertRaisesRegex(RuntimeError, "DeepSeek text embedding API is reserved"):
            deepseek.encode_text(["hello"])

    def test_text_embedding_provider_factory_prefers_local_without_api_calls(self):
        local = build_text_embedding_provider(EmbeddingConfig(enable_api_calls=False, text_dim=6))
        qwen = build_text_embedding_provider(EmbeddingConfig(enable_api_calls=True, provider_order=("qwen",), text_dim=6))
        deepseek = build_text_embedding_provider(
            EmbeddingConfig(enable_api_calls=True, provider_order=("deepseek",), text_dim=6)
        )

        self.assertIsInstance(local, DeterministicHashTextEmbeddingProvider)
        self.assertIsInstance(qwen, QwenTextEmbeddingProvider)
        self.assertIsInstance(deepseek, DeepSeekTextEmbeddingProvider)


if __name__ == "__main__":
    unittest.main()
