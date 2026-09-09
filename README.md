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

`QwenTextEmbeddingProvider` and `DeepSeekTextEmbeddingProvider` reserve the future API integration points. Their credentials are expected from `DASHSCOPE_API_KEY` and `DEEPSEEK_API_KEY`; secrets must not be written into configuration files. The current reserved implementations raise a clear error instead of making a network request. When the concrete SDK clients are added, they should preserve the existing `encode_text(...) -> torch.Tensor` interface and write results through `CachedEmbeddingStore` before training consumes them.

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

## Verify

```bash
python -m unittest discover -s tests -v
```

## Run Synthetic Experiment

```bash
python experiments/run_synthetic.py
```
