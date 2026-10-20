# SIR-CDR: Text-only Cross-domain Recommendation

## Optional Qwen user-side latent reasoning

The `llm-cdr-v1` path reuses the Qwen3-Embedding-8B assets. The active Sports/Clothing
and Phones/Electronics templates use a pinned Qwen3-0.6B backbone with LoRA,
projected structural tokens, and two latent passes with separate per-example
memory carried between the passes. Explicit `llm_*_4b_v1.yaml` templates preserve
the previous 4B configuration. This path has its own runner/configs and
compact checkpoints; the Text v4 baseline below is unchanged. It is not a
pure-generative ranking migration or an AGCLR CoT-curriculum reproduction.
See [the Chinese implementation and server guide](docs/llm_latent_v1_zh.md).
For the 0.6B migration and isolated server commands, see
[the small-model guide](docs/qwen06_user_backbone_zh.md).


The active v4 pipeline combines cached text embeddings, sparse shared/private graphs,
gated latent reasoning, context prediction feedback (CPF), and Semantic ID generation.
It accepts no image or multimodal inputs and learns no raw item-ID embeddings.
Legacy synthetic and v1 modules remain available only for reproducing earlier runs.

See [the v2 mechanism and server guide](docs/text_cdr_v2.md) for implementation
scope, inference modes, assumptions, ablations and known experimental limits.

For a file-by-file, function-by-function Chinese walkthrough of environment setup,
data preparation, model internals, training calls and evaluation, see the
[complete code and experiment guide](docs/complete_code_guide_zh.md).

```bash
python experiments/run_text_cdr.py --config configs/text_sports_clothing_v4_p0.yaml --action train --variant full --seeds 42 --device cuda
```

This uses a separate output directory and requires new embedding, tokenizer and
recommender training with the local `Qwen/Qwen3-Embedding-8B` backbone.
Performance improvements require validation on real data; unit tests establish
correctness, not recommendation quality.

The first real-data targets are:

- Amazon: Sports-Clothing and Phones-Electronics in both transfer directions.
- Douban: Books-Movies in both transfer directions.

## Documentation

- [Amazon Sports-to-Clothing preprocessing guide](docs/amazon_sports_clothing_preprocessing.md)
- [Qwen3-Embedding-8B migration and server guide](docs/qwen3_embedding_8b.md)
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
- `cdr_framework/modules.py`: multimodal projection, confidence-aware graph propagation, shared/specific disentanglement (user and prototype sides), cross-domain/specific structural injectors, codebook-summary pooling, and gated latent reasoning.
- `cdr_framework/tokenization/`: domain-adaptive semantic tokenizer, semantic codebook, beam search, and item-token lookup.
- `cdr_framework/modules_cpf.py`: context prediction feedback head and its prediction/alignment/anti-collapse objectives.
- `cdr_framework/metrics.py`: HR@K, NDCG@K, MRR@K, catalog coverage, and token lookup hit rate.
- `cdr_framework/framework.py`: runnable end-to-end synthetic baseline that preserves the existing model path.

The active text-only model (`cdr_framework/text_model.py`) implements the dual
structural injection stage on top of the v2 pipeline: item-side prototype
disentanglement (shared/source-specific/target-specific prototypes from the
sparse graph outputs), CrossDomain/SpecificDomain structural injectors that
feed the shared and private latent paths of the gated reasoner, and a
codebook-summary prefix. Regularizers are contrastive shared alignment,
prototype orthogonality, and source-private separation. Every new component is
controlled by a config switch, so `v2_legacy` reproduces the previous
single-gate wiring and the new ablations (`no_proto`, `no_cd_inj`, `no_sp_inj`,
`no_lsep`, `no_lsh`, `no_csum`) isolate each contribution.

The intended data flow is:

```text
interactions + metadata
  -> cached multimodal embeddings
  -> source/target/cross-domain graphs
  -> shared and domain-specific structural representations
  -> prototype-side shared/specific item tokens
  -> domain-adaptive semantic tokens
  -> cross-domain/specific structural injectors
  -> gated latent reasoning
  -> CPF training feedback
  -> dual structural fusion prefix (with codebook summary)
  -> autoregressive token generation and item lookup
```

## Embedding Backbone Boundary

External embedding calls are disabled by default. `EmbeddingConfig(enable_api_calls=False)` selects the deterministic local provider, so tests and synthetic experiments do not require network access or credentials.

The active text configuration uses the official local `Qwen/Qwen3-Embedding-8B`
weights in BF16 and exports normalized 768-dimensional MRL vectors. Set
`QWEN3_EMBEDDING_MODEL_PATH` to a downloaded model directory; if it is unset,
Sentence Transformers downloads the public Hugging Face model. The legacy
`QwenTextEmbeddingProvider` remains available for reproducing the earlier
OpenAI-compatible `text-embedding-v4` run.

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

After preprocessing, download the 8B weights and verify one local embedding:

```bash
export QWEN3_EMBEDDING_MODEL_PATH=/root/autodl-tmp/models/Qwen3-Embedding-8B
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

The tokenizer first learns domain-adaptive continuous item representations and then fits one data-driven residual K-means codebook per Semantic ID position. A run is marked complete only when its collision rate and every level's codebook utilization pass the configured quality gates. This stage does not call any external API. It exports fixed-length Semantic IDs, independent residual codebooks, item latent vectors, training history, and quality statistics under `artifacts/tokenizer/qwen3_8b/sports_to_clothing/`.

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

For the text-only v4 model, tune retrieval/generation fusion on validation
without retraining or touching test data:

```bash
python experiments/sweep_text_fusion.py \
  --config configs/text_sports_clothing_v4_p0.yaml \
  --variant full --seed 42 --device cuda \
  --weights 0 0.1 0.25 0.5 1 2 \
  --fusion-norm log_softmax
```

After selecting the score weight on validation, update only the score fields in
the v4 YAML and run the held-out test exactly once:

```bash
python experiments/run_text_cdr.py \
  --config configs/text_sports_clothing_v4_p0.yaml \
  --action evaluate --variant full --seeds 42 \
  --split test --mode hybrid --device cuda
```

The v4 model changes learned parameter wiring and checkpoint identity. A v3
`best_model.pt` cannot be evaluated with v4 code and must not be copied into
`artifacts/text_cdr_v4/`; train a new v4 checkpoint in that isolated output root.

## Active Dataset Tasks and GenCDR Douban Alignment

The comparison baseline is [GenCDR](https://github.com/hupeiyu21/GenCDR), not
GenCDSR. The default runners select only the two Amazon tasks while the original
GenCDR Douban population and evaluation protocol are being verified. The third
registry entry is exclusively `douban_book_movie`; the old GenCDSR third-task
adapter, configs and local data have been removed.
See [GenCDR Douban source and alignment audit](docs/gencdr_douban_alignment_zh.md).
The [asset design and comparison acceptance criteria](docs/douban_book_movie_asset_design_zh.md)
describe the remaining stages and why existing results are not yet a matched
comparison with the published GenCDR numbers.
The legacy `run_gencdr_benchmarks.py` trains Text v4; LLM v1 uses `run_llm_cdr.py`.

- Sports -> Clothing
- Phones -> Electronics
- GenCDR Douban Books/Movies: raw assets available; exact alignment pending

The runner requires completed processed splits and tokenizer schema v2 artifacts
for every selected pair. It checks every input before allocating the GPU and
stops without starting any training when an input is missing or invalid.

```bash
python experiments/run_gencdr_benchmarks.py --action preflight
```

If the preflight reports missing pair assets, prepare every missing stage with
one resumable command. Completed pairs and completed stages are validated and
skipped. Raw Amazon 2014 gzip files are downloaded only when they are absent;
Qwen embedding batches and tokenizer epochs resume from their managed outputs.

```bash
mkdir -p logs
nohup python -u experiments/prepare_gencdr_assets.py \
  --pairs phones_to_electronics --device cuda \
  > logs/prepare_gencdr_assets.log 2>&1 &
echo $! > logs/prepare_gencdr_assets.pid
```

The Amazon tasks retain their existing processing and asset locations. Prepare
and audit the original Douban source independently, without allocating the GPU:

```bash
python experiments/prepare_gencdr_douban.py --action all
```

This downloads a hash-pinned publisher archive and reports discrepancies with
GenCDR Table 1. It does not create SIR training splits or claim exact paper
alignment. A two-task preflight success does not certify the third task. The
removed GenCDSR task and Amazon Books/Movies are not substitutes.

Run the two active seed-42 jobs sequentially on one GPU. Training is resumable. After
each checkpoint is complete, the runner tunes score fusion on validation for
`none`, `log_softmax`, and `zscore`, then evaluates the held-out test split once.

```bash
mkdir -p logs
nohup python -u experiments/run_gencdr_benchmarks.py \
  --action all --device cuda --seed 42 \
  > logs/gencdr_three_datasets_seed42.log 2>&1 &
echo $! > logs/gencdr_three_datasets_seed42.pid
```

Monitor and summarize:

```bash
tail -f logs/gencdr_three_datasets_seed42.log
python experiments/run_gencdr_benchmarks.py --action summarize --seed 42
```

Final reports are written to:

```text
artifacts/text_cdr_v4/benchmark_reports/gencdr_three_datasets_seed42.json
artifacts/text_cdr_v4/benchmark_reports/gencdr_three_datasets_seed42.csv
```

The report filenames above are legacy names; inspect the actual dataset keys.
The active run is one directional. Reporting both target domains requires
separately configured reverse-direction runs.

## Faster Qwen3-0.6B Experiments

Use `python experiments/run_llm_fast.py --action prepare`, then
`python -u experiments/run_llm_fast.py --action run` for serial Phones/Electronics
and Sports/Clothing training, validation tuning, verification and final testing.
This is an independent batch-4 experiment (accumulation 4, eval batch 4,
checkpointing off), with evaluation catalog reuse and device-side loss logging.
All settings/code remain signed; old checkpoints are preserved and not silently
resumed. See [server commands and scientific boundaries](docs/llm_efficiency_zh.md).
Final evaluation now uses a checkpoint-bound CPU catalog snapshot and FP32 GPU
scoring, with exact independent validation-score verification before testing.
Completed training is reused. Old native scoring reports remain available.
See [evaluation recovery and disclosure](docs/llm_snapshot_evaluation_zh.md).

## Third task: Douban Book to Movie

Use `python -u experiments/run_douban.py --action all` to prepare the verified
Douban archive, generate 8B embeddings and semantic IDs, then train Qwen3-0.6B
and run strictly verified validation/test evaluation. Splitting follows the two
Amazon tasks, including duplicate retention and temporal source visibility.
This is a common-user protocol; authors' final GenCDR split alignment remains
unverified. See [Douban server workflow](docs/douban_execution_zh.md).

## Reverse-direction assets

Reverse assets can be prepared independently while forward training continues:
`python -u experiments/prepare_reverse_assets.py --action cpu --workers 3 --cpu-tokenizer`.
GPU completion is a separate serial stage that refuses a busy GPU. See
[reverse asset preparation](docs/reverse_assets_parallel_zh.md).

## Verify

```bash
python -m unittest discover -s tests -v
```

## Run Synthetic Experiment

```bash
python experiments/run_synthetic.py
```

## GenCDR shared-data baseline entry

For new, isolated leave-one-out assets and Recall@10 checkpoint selection, see
[GenCDR shared protocol (Chinese)](docs/gencdr_shared_protocol_zh.md).
Run `python experiments/run_gencdr_shared.py --action convert`, then `--action prepare`,
then `--action run`. This is a common-protocol reproduction, not a verified copy
of the author's unavailable final splits. Existing experiment entry points retain their original assets.
