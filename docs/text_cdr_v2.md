# Text-only SIR-CDR v2

## Status and scope

The active v2 pipeline accepts cached text-derived latent vectors and Semantic
IDs only. No image, multimodal, learned item-ID embedding, or online LLM input
is used. Legacy synthetic modules remain for reproducibility; do not use them
to describe the v2 experiment. The active configuration uses local
Qwen3-Embedding-8B vectors and therefore requires fresh embedding, tokenizer
and recommendation artifacts; legacy text-embedding-v4 artifacts are not reused.

Implemented stages: training-only sparse shared/source/target graphs; gated
structural injection; source and target sequence GRUs with masked attention;
shared/private user heads and selective source transfer; gated implicit state
updates; iterative CPF prediction feedback; fixed-length autoregressive Semantic
ID supervision; full-target retrieval loss; retrieval, hybrid, exhaustive
generation-scored ranking and catalog-constrained generation; early stopping,
learning-rate scheduling, resume, ablations and multi-seed validation summaries.

This is a local neural recommender, not an LLM decoder fine-tuning experiment.
It implements the project's module roles, not a verified reproduction of
GenCDR, AGCLR or LT-Tuning. Metrics on remote real data remain to be measured.
Additional raw-data adapters for Phones/Electronics and Douban are not supplied
by this update; the tested input contract is the existing Sports/Clothing pair.

## Continuous gated implicit reasoning

Let c denote the sequence context, g the shared cross-domain signal, and p the
target-private signal. Initialization and each internal step follow:

```text
h0 = tanh(Winit [c; g; p] + binit)
zt = [ht; c; g; p]
read = sigmoid(Wread zt + bread)
write = sigmoid(Wwrite zt + bwrite)
forget = sigmoid(Wforget zt + bforget)
candidate = tanh(Wcandidate [read * ht; c; g; p] + bcandidate)
h(t+1) = LayerNorm(forget * ht + write * candidate)
```

Read controls the previous state entering the candidate transformation. Write
controls new information, and forget controls retention of the old state. These
are learned element-wise soft gates, not discrete rules or verbal reasoning.
The same parameters are reused across steps. Shared and private heads project
the final state together with their corresponding structural signals.

The v2 outer feedback loop computes a CPF semantic prediction from these heads
and c, then updates c through a learned gate before the next reasoning round.
Default: two feedback rounds, each containing three internal gated steps. The
state is reinitialized from the updated context at the next outer round;
this is not an uninterrupted six-step recurrence. Future-item supervision is
used only for training losses. Inference uses history and model predictions.

Why no LLM call: gates and state updates are small trainable PyTorch layers.
Qwen contributes upstream cached semantic embeddings, but it is not invoked
inside this reasoning block. Semantic ID generation is performed by a trained
GRU decoder, not Qwen/DeepSeek. Calling these latent updates "implicit reasoning"
does not by itself establish reasoning ability or performance improvement;
the no_reasoning, single_step and no_feedback variants test those claims.

## Structure, protocols and generation

Graphs use each user's latest training history prefix once. Domain edges join
adjacent history items; cross edges join the last five source and target items
as co-user links, not temporal transitions. Positive labels never add graph
edges. Graphs include non-padding self loops and symmetric degree normalization,
and propagation uses sparse multiplication with learned path projections.

This is a static training-graph protocol under per-user leave-one-out splitting,
not strict global chronological online forecasting. Training interactions from
other users or later training prefixes may contribute graph context. No
validation/test interaction builds graph edges. Catalog text/tokenization was
fit over the known catalog; this is a transductive catalog assumption.

Source/target histories are truncated to 50 for encoding; seen-item filtering
uses the FULL target history. No true-label exception is made. If a repeated
positive is already seen, it remains a miss in the metric denominator. This
protocol differs from the legacy truncated-history filter. Raw item IDs index
text features and labels only; Semantic ID tokens are text-derived codes.
Each residual level receives its own vocabulary offset (level * codebook_size).

Modes:

- retrieval: score every target item with the semantic query.
- hybrid: full-target retrieval then generation-score reranking, default pool 500.
- exhaustive: score every target candidate's complete code sequence and EOS,
  fuse with retrieval scores, then take exact top-K. Decoder batches are bounded.
- generate: beam search over legal target-catalog token prefixes, include EOS
  score and resolve same-code collisions by query-item cosine. No retrieval
  fallback. Unfilled slots are zero and count as misses, never as item hits.

All modes exclude seen items, do not sample random negatives, and use no labels
to rank. Beam search remains approximate unless it covers all relevant prefixes.
Exhaustive scoring can be much slower than hybrid. These modes must be named in
reported experimental results rather than treated as the same protocol.

## Run on the server

Upload the provided update folder contents to the repository root, then:

```bash
cd /root/autodl-tmp/SIR-CDR
git pull --ff-only origin main
mkdir -p logs
nohup python -u experiments/run_text_cdr.py --config configs/text_sports_clothing.yaml --action train --variant full --seeds 42 --device cuda > logs/text_cdr_full.log 2>&1 &
echo $! > logs/text_cdr_full.pid
tail -f logs/text_cdr_full.log
```

Outputs are under `artifacts/text_cdr/qwen3_8b/sports_to_clothing/full_seed42`, separate
from the original recommender. Lower LR 0.0003, dropout 0.1, weight decay 0.0001
and ReduceLROnPlateau aim to improve generalization, without promising a gain.
Epoch 1 is evaluated, then every 2 epochs; early stopping uses NDCG@10 and eight
consecutive non-improving validations. Checkpoints include optimizer, scheduler,
CPU/CUDA RNG, input and code hashes. Restart the same command to resume. If code
or training config changes, choose a new --output-root, never overwrite baseline.

After training, compare validation inference modes:

```bash
python -u experiments/run_text_cdr.py --action evaluate --split validation --mode retrieval --device cuda
python -u experiments/run_text_cdr.py --action evaluate --split validation --mode hybrid --device cuda
python -u experiments/run_text_cdr.py --action evaluate --split validation --mode generate --device cuda
python -u experiments/run_text_cdr.py --action evaluate --split validation --mode exhaustive --device cuda
```

The default configuration path is resolved from the repository. Specify it
explicitly if using a custom YAML. Mode overrides are evaluation-only and
stored in distinct reports. The checkpoint was selected using the configured
validation mode, which must be reported when comparing other inference modes.

### Validation-only fusion sweep

After confirming that the configured candidate pool matches exhaustive
validation at the reported cutoffs, tune only the generation contribution:

```bash
python -u experiments/sweep_text_fusion.py \
  --config configs/text_sports_clothing.yaml \
  --variant full --seed 42 --device cuda \
  --weights 0 0.1 0.25 0.5 1 2 \
  --batch-size 16
```

The command loads the unchanged best checkpoint, evaluates only the validation
split, and selects by `NDCG@10`. It never updates model parameters or reads the
test split. Reports are fingerprinted and written beside the checkpoint as
`validation_text_fusion_sweep_<hash>.json`; rerunning the same command skips
completed weights. Exact ties prefer the smaller generation weight.

## Ablation suite

```bash
python -u experiments/run_text_cdr.py --action suite --variants full no_graph no_reasoning single_step no_feedback no_semantic no_cpf_loss target_only --seeds 42 43 44 --device cuda
python experiments/summarize_text_cdr.py --root artifacts/text_cdr/qwen3_8b/sports_to_clothing
```

This launches up to 24 sequential training runs; existing completed runs with
matching signatures are reused. no_semantic removes code features but retains
code supervision, so it is not a removal of the entire generation task.
no_cpf_loss removes only the auxiliary objective, retaining feedback and the
query head. no_feedback reduces the outer loop to one round. target_only
disables source histories and graphs; it is not a reproduction of a named
single-domain baseline. Exact algorithmic comparisons with external baselines
still require their implementations under the same split and evaluation rules.

Select settings using validation. Only after final selection, run the intended
test protocol, for example:

```bash
python -u experiments/run_text_cdr.py --action evaluate --split test --mode hybrid --device cuda
```

Do not infer improvements from CPU fixture metrics. Remote 3090 training and
real validation results determine whether the new components improve R@5/N@5/
R@10/N@10. API keys, raw datasets, tokenizer weights, logs and model checkpoints
must stay outside the GitHub upload bundle.
