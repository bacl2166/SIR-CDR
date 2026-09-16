# Qwen3-Embedding-8B Migration and Server Run Guide

The active SIR-CDR v2 embedding backbone is the official open-source
`Qwen/Qwen3-Embedding-8B`. It runs locally in BF16 on CUDA, uses left padding
and PyTorch SDPA, truncates inputs to 8192 tokens, and exports normalized
768-dimensional Matryoshka embeddings. Raw item IDs remain indices only.

The previous `text-embedding-v4` API implementation remains in the repository
for legacy reproduction, but the active YAML does not read DashScope keys.

## Artifact isolation

The 8B pipeline writes to new directories:

```text
artifacts/embeddings/qwen3_8b/sports_to_clothing
artifacts/tokenizer/qwen3_8b/sports_to_clothing
artifacts/recommendation/qwen3_8b/sports_to_clothing
artifacts/text_cdr/qwen3_8b/sports_to_clothing
```

Do not copy old `text-embedding-v4` chunks into these directories. Changing the
backbone changes every vector, so the tokenizer and recommender must be trained
again.

## Download on the remote server

The model is public and does not require a DashScope API key. Download it once:

```bash
cd /root/autodl-tmp/SIR-CDR
conda activate sir-cdr

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-8B",
    local_dir="/root/autodl-tmp/models/Qwen3-Embedding-8B",
)
PY

export QWEN3_EMBEDDING_MODEL_PATH=/root/autodl-tmp/models/Qwen3-Embedding-8B
```

Persist only the path, not credentials:

```bash
printf '%s\n' \
  'export QWEN3_EMBEDDING_MODEL_PATH=/root/autodl-tmp/models/Qwen3-Embedding-8B' \
  > /root/autodl-tmp/sir-cdr-qwen3.env
source /root/autodl-tmp/sir-cdr-qwen3.env
```

## Smoke test and full embedding job

The smoke test loads the 8B model and embeds one item:

```bash
python experiments/embed_items.py \
  --config configs/amazon_sports_clothing.yaml \
  --smoke-test
```

Expected identity:

```text
Provider: qwen3_local
Model: Qwen/Qwen3-Embedding-8B
Dimension: 768
```

Start the resumable full job:

```bash
mkdir -p logs
nohup python -u experiments/embed_items.py \
  --config configs/amazon_sports_clothing.yaml \
  > logs/embed_qwen3_8b.log 2>&1 &
echo $! > logs/embed_qwen3_8b.pid
tail -f logs/embed_qwen3_8b.log
```

Batch size is one for the 24 GB RTX 3090. If CUDA runs out of memory on an
unusually long product record, reduce `max_sequence_length` in the YAML, choose
a new embedding output directory, and restart. Do not mix chunks generated with
different sequence limits; the progress identity rejects that mismatch.

## Rebuild downstream artifacts

```bash
nohup python -u experiments/train_tokenizer.py \
  --config configs/amazon_sports_clothing.yaml \
  --device cuda \
  > logs/train_tokenizer_qwen3_8b.log 2>&1 &

nohup python -u experiments/run_text_cdr.py \
  --config configs/text_sports_clothing.yaml \
  --action train --variant full --seeds 42 --device cuda \
  > logs/text_cdr_qwen3_8b_full_seed42.log 2>&1 &
```

Model weights, generated vectors, checkpoints and logs stay outside Git. Only
the provider, configurations, tests and documentation are version controlled.
