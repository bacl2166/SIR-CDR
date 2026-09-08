"""Dual structural injection framework for generative cross-domain recommendation."""

from cdr_framework import datasets, embeddings, graphs, tokenization
from cdr_framework.config import DatasetConfig, EmbeddingConfig, FrameworkConfig, ModelConfig, TrainingConfig
from cdr_framework.framework import DualStructuralInjectionRecommender
from cdr_framework.metrics import coverage, hit_rate_at_k, mrr_at_k, ndcg_at_k, token_lookup_hit_rate
from cdr_framework.modules_cpf import ContextPredictionFeedback

__all__ = [
    "ContextPredictionFeedback",
    "DatasetConfig",
    "DualStructuralInjectionRecommender",
    "EmbeddingConfig",
    "FrameworkConfig",
    "ModelConfig",
    "TrainingConfig",
    "coverage",
    "datasets",
    "embeddings",
    "graphs",
    "hit_rate_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "token_lookup_hit_rate",
    "tokenization",
]
