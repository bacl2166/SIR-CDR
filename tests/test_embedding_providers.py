import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch

from cdr_framework.config import EmbeddingConfig
from cdr_framework.embeddings.api import DeepSeekTextEmbeddingProvider, QwenTextEmbeddingProvider, build_text_embedding_provider
from cdr_framework.embeddings.cache import CachedEmbeddingStore
from cdr_framework.embeddings.local import DeterministicHashTextEmbeddingProvider
from cdr_framework.embeddings.providers import BaseLLMSemanticProvider


class EmbeddingProviderTest(unittest.TestCase):
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
