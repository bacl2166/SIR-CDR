# Dual Structural Injection Generative CDR Framework

This repository contains a research framework for generative cross-domain recommendation. The implementation combines GenCDR-inspired multimodal embeddings, confidence-aware graph structure, AGCLR-inspired gated latent reasoning, semantic token generation, and LT-Tuning-inspired context prediction feedback (CPF).

The first real-data targets are:

- Amazon: Sports-Clothing and Phones-Electronics in both transfer directions.
- Douban: Books-Movies in both transfer directions.

## Documentation

- [Amazon Sports-to-Clothing preprocessing guide](docs/amazon_sports_clothing_preprocessing.md)
- [Amazon preprocessing implementation plan](docs/superpowers/plans/2026-09-09-amazon-sports-clothing-preprocessing.md)

## Data Preparation Status

The Amazon Sports-to-Clothing data contract, leakage controls, artifact schemas, server commands, and acceptance checks are documented. The executable preprocessing stage is available through `experiments/prepare_amazon.py`.

The currently executable checks are:

```bash
python -m unittest discover -s tests -v
python experiments/run_synthetic.py
python experiments/prepare_amazon.py --help
```

Raw Amazon Reviews 2014 files and all generated outputs belong under ignored `data/` and `artifacts/` directories. Product ASIN values are reserved for indexing, supervision, and reverse lookup; they are not model input features.

Prepare or download the Amazon Reviews 2014 Sports-to-Clothing data:

```bash
python experiments/prepare_amazon.py \
  --config configs/amazon_sports_clothing.yaml
```

Use `--skip-download` when all four raw gzip files already exist, and use `--force` only when intentionally replacing a completed processed directory. This stage does not call an embedding API.

## Framework Modules

- `cdr_framework/config.py`: dataset, embedding, model, and training configuration dataclasses.
- `cdr_framework/datasets/`: Amazon/Douban-compatible interaction and metadata schemas plus chronological leave-one-out splitting.
- `cdr_framework/embeddings/`: text/image/LLM provider boundaries, deterministic offline embeddings, filesystem cache, and reserved Qwen/DeepSeek API providers.
- `cdr_framework/graphs/`: transition graph construction and seven-dimensional edge confidence features.
- `cdr_framework/modules.py`: multimodal projection, confidence-aware graph propagation, shared/specific disentanglement, structural injection, and gated latent reasoning.
- `cdr_framework/tokenization/`: domain-adaptive semantic tokenizer, semantic codebook, beam search, and item-token lookup.
- `cdr_framework/modules_cpf.py`: context prediction feedback head and its prediction/alignment/anti-collapse objectives.
- `cdr_framework/metrics.py`: HR@K, NDCG@K, MRR@K, catalog coverage, and token lookup hit rate.
- `cdr_framework/framework.py`: runnable end-to-end synthetic baseline that preserves the existing model path.

The intended data flow is:

```text
interactions + metadata
  -> cached multimodal embeddings
  -> source/target/cross-domain graphs
  -> shared and domain-specific structural representations
  -> domain-adaptive semantic tokens
  -> gated latent reasoning
  -> CPF training feedback
  -> autoregressive token generation and item lookup
```

## Embedding API Boundary

External embedding calls are disabled by default. `EmbeddingConfig(enable_api_calls=False)` selects the deterministic local provider, so tests and synthetic experiments do not require network access or credentials.

`QwenTextEmbeddingProvider` implements the OpenAI-compatible `text-embedding-v4` call and reads credentials from `DASHSCOPE_API_KEY` and `DASHSCOPE_BASE_URL`. `DeepSeekTextEmbeddingProvider` remains a reserved fallback. Secrets must not be written into configuration files. Qwen outputs preserve the existing `encode_text(...) -> torch.Tensor` interface and the batch runner persists completed chunks before training consumes them.

Provider selection is centralized:

```python
from cdr_framework.config import EmbeddingConfig
from cdr_framework.embeddings import build_text_embedding_provider

provider = build_text_embedding_provider(
    EmbeddingConfig(
        enable_api_calls=False,
        provider_order=("qwen", "deepseek", "local"),
    )
)
```

After preprocessing, verify one real Qwen request before starting the resumable full job:

```bash
source /root/autodl-tmp/sir-cdr-api.env
python experiments/embed_items.py \
  --config configs/amazon_sports_clothing.yaml \
  --smoke-test

python experiments/embed_items.py \
  --config configs/amazon_sports_clothing.yaml
```

Train the domain-adaptive tokenizer after the embedding manifest and tensor have been verified:

```bash
python experiments/train_tokenizer.py \
  --config configs/amazon_sports_clothing.yaml \
  --device cuda
```

The tokenizer first learns domain-adaptive continuous item representations and then fits one data-driven residual K-means codebook per Semantic ID position. A run is marked complete only when its collision rate and every level's codebook utilization pass the configured quality gates. This stage does not call any external API. It exports fixed-length Semantic IDs, independent residual codebooks, item latent vectors, training history, and quality statistics under `artifacts/tokenizer/sports_to_clothing/`.

Outputs created by the obsolete shared-codebook schema must be restarted once:

```bash
python experiments/train_tokenizer.py \
  --config configs/amazon_sports_clothing.yaml \
  --device cuda \
  --force
```

## Formal Recommendation Training

After tokenizer schema v2 passes its quality gate, train SIR-CDR with a full-target softmax objective:

```bash
python experiments/train_recommender.py \
  --config configs/amazon_sports_clothing.yaml \
  --device cuda
```

The formal model consumes fixed text-derived item latents and Semantic IDs. Numeric item IDs are used only to retrieve these fixed features, identify labels, and filter seen target items. Training does not use randomly sampled negatives. Validation and test ranking score the complete target catalog, then apply Semantic ID generation reranking to the strongest full-catalog candidates.

Evaluate the best validation checkpoint once on the held-out test set:

```bash
python experiments/evaluate_recommender.py \
  --config configs/amazon_sports_clothing.yaml \
  --device cuda \
  --split test
```

## Verify

```bash
python -m unittest discover -s tests -v
```

## Run Synthetic Experiment

```bash
python experiments/run_synthetic.py
```
