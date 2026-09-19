# SIR-CDR Architecture-Preserving Performance Improvement Design

**Status:** Draft for user review; implementation not started  
**Date:** 2026-09-20  
**Target:** Amazon Sports -> Clothing text-only SIR-CDR pipeline  
**Primary metric:** validation NDCG@10  
**Secondary metrics:** validation HR@10, NDCG@5, HR@5, MRR@10, catalog coverage, latency

## 1. Objective

Improve recommendation quality while preserving the defining SIR-CDR architecture:

- text-derived item representations and Semantic IDs;
- training-only shared/source/target graphs;
- shared and domain-specific user interests;
- item prototypes and CD/SP structural injection;
- iterative implicit reasoning with CPF feedback;
- joint retrieval and Semantic-ID autoregressive decoding;
- full-catalog history-masked evaluation.

This work does not attempt to turn SIR-CDR into GenCDR or replace its lightweight
reasoning model with an LLM recommender. GenCDR numbers may be contextual references,
but all claims about improvement will be relative to a frozen SIR-CDR baseline under
the same data, splits, catalog, masking, and metric implementation.

## 2. Experimental Integrity

The existing v3 test result is an observed pilot baseline and must not be used for
model or hyperparameter selection. All development decisions use the validation split.
The final test split is evaluated once after the architecture, inference rule, and
hyperparameters are frozen. Reports must disclose that the baseline test result was
observed before development began.

Every experiment records:

- Git commit and dirty-worktree state;
- complete resolved configuration;
- processed-data, embedding, tokenizer, and checkpoint SHA256 identities;
- random seed, device, PyTorch/CUDA versions, and elapsed time;
- per-user ranks or top-k lists needed for paired analysis;
- all primary and secondary metrics, including unsuccessful runs.

Seed 42 may be used for bounded screening. Any promoted configuration is rerun with
seeds 42, 43, and 44, alongside the unchanged baseline. Final claims report mean,
sample standard deviation, relative change, and a paired user-level bootstrap 95%
confidence interval. The search space and primary metric are fixed before screening.

## 3. Versioning and Compatibility

Architecture-semantic changes create `text-cdr-v4`; v3 checkpoints are never relabeled
or silently loaded as v4. Each stage writes to a new output root. Existing v2/v3
artifacts remain immutable.

Configuration is divided into two identities:

1. **Training identity:** fields that affect learned parameters or training data.
2. **Evaluation identity:** checkpoint hash plus all runtime ranking and fusion fields.

Inference-only settings must be explicit dataclass fields and included in evaluation
reports. They are excluded from training identity only when they provably cannot change
the trained weights. `--allow-code-mismatch` remains limited to inference tooling drift;
it never bypasses version, training-config, data, or tokenizer identity checks.

## 4. Phase P0: Correctness and Reproducibility

### 4.1 Masked structural aggregation

Current CD attention and SP means include padded positions. P0 adds reusable masked
operations driven by `source_lengths` and `target_lengths`:

- `masked_sequence_mean(values, ids, lengths)` averages only valid positions;
- `masked_attention_pool(values, ids, lengths, query)` assigns padded positions
  `-inf` before softmax;
- every batch is required to have positive source and target lengths;
- padding output remains exactly zero after projections and pooling.

`CrossDomainStructuralInjector.forward` receives both length tensors, constructs a
concatenated validity mask, and performs masked attention. `SpecificDomainStructuralInjector.forward`
receives the lengths and performs separate masked means. Results must be invariant to
additional right padding.

### 4.2 Symmetric private-interest fusion

The full v3 path computes sequence-private representations but replaces them with SP
structural signals. V4 retains both through learnable gates:

```text
g_S = sigmoid(W_S [source_private ; source_sp])
z_S,sp = LayerNorm(g_S * source_private + (1-g_S) * source_sp)

g_T = sigmoid(W_T [target_private ; target_sp])
z_T,sp = LayerNorm(g_T * target_private + (1-g_T) * target_sp)
```

The fused target-specific representation is supplied to the implicit reasoner and
prefix generator. The fused source-specific representation participates in separation
regularization and diagnostics. When SP injection is disabled, the sequence-private
representations pass through unchanged. Tests must confirm nonzero gradients reach
`source_head`, `target_head`, and both structural injector branches in the full model.

### 4.3 Separate training and inference weights

The following fields have distinct meanings:

- `retrieval_loss_weight` and `generation_loss_weight` compose the training objective;
- `retrieval_score_weight` and `generation_score_weight` compose inference scores.

Legacy `retrieval_weight` and `generation_weight` remain accepted only through a
documented migration path and may not ambiguously control both phases. Default training
weights are 1.0/1.0, preserving the v3 objective before deliberate tuning.

### 4.4 Explicit score calibration

Hybrid ranking supports explicit, recorded calibration modes:

- `none`: v3-compatible raw score fusion;
- `log_softmax`: masked candidate-wise log-probability calibration;
- `zscore`: masked per-user standardization with an epsilon floor.

Calibration is applied only to finite eligible candidates. It must never unmask seen
items or create NaN/Inf values. Candidate count, retrieval temperature, calibration
mode, and score weights form the evaluation identity. Rank fusion is deferred until
these calibrated score methods have been evaluated.

## 5. Phase P1: Target-Aware Catalog Scoring

V3 builds a structurally rich user query but compares it with base item vectors. V4
adds a target-aware catalog representation using existing SIR signals:

```text
g_graph = sigmoid(W_graph [base ; graph_target])
g_proto = sigmoid(W_proto [base ; proto_target])

target_item = LayerNorm(
    base
    + g_graph * graph_target
    + g_proto * proto_target
)
```

If prototypes are disabled, the prototype term is zero. If graphs are disabled, both
structural terms are zero and `target_item` reduces to normalized base content. Padding
row zero is restored after fusion. Retrieval loss, candidate generation, and every
inference mode use the same target catalog tensor.

Required ablations are:

1. base only;
2. base plus target graph;
3. base plus target graph and target prototype.

This phase does not change the user reasoner, graph construction, or SID decoder.

## 6. Phase P2: Training Stability

P2 is attempted only after P0 and P1 are independently measured.

### 6.1 Centralized loss composition

The model returns raw differentiable loss components. The training controller composes
the total loss using the resolved schedule and records the effective coefficient of
every component each epoch. Retrieval and generation remain active from epoch one.
CPF, alignment, separation, LSEP, and prototype orthogonality use a linear warm-up:

```text
aux_scale(epoch) = min(1, epoch / aux_warmup_epochs)
```

The baseline comparison uses `aux_warmup_epochs=0`; warm-up is a separate experiment.

### 6.2 User-balanced training sampler

Sliding target prefixes give long-history users more rows. An optional deterministic
sampler assigns each row weight `1 / rows_for_user`, draws exactly `len(train_rows)`
rows per epoch with replacement, and uses `seed + epoch`. Uniform row sampling remains
the baseline. The sampler choice is part of training identity.

### 6.3 Controlled capacity experiments

After correctness fixes, vary one factor at a time:

- hidden dimension 128 versus 256;
- reasoning steps 3 versus 5;
- feedback rounds 2 versus 3;
- training budget/patience;
- loss weights selected from a small preregistered grid.

Combined configurations are created only from factors that improve validation results
individually. Existing mixed configurations are not treated as causal evidence.

### 6.4 Optional weighted static graphs

If graph ablations show that graph-enabled v4 is not better than `no_graph`, retain the
same training-only static graph topology but compare binary edges with normalized
log-frequency and recency-weighted edges. Validation/test events remain excluded. This
is an optional later experiment, not part of P0 or P1.

## 7. Phase P3: Semantic-ID Quality

P3 begins only if diagnostics indicate that generation or SID structure is a bottleneck.
Before changing tokenization, record:

- collision rate and unique SID count;
- utilization and entropy at every codebook level;
- quantization MSE;
- duplicate-prefix counts by depth;
- correlation between embedding similarity and SID common-prefix length;
- generation validity and oracle prefix-tree recall.

Initial controlled alternatives are `4 x 512` and `3 x 1024`, with the same Qwen3
embeddings and train/validation/test data. If quantization-aware fine-tuning is later
introduced, it receives a new tokenizer schema and new downstream output roots. No
tokenizer is chosen using recommendation test metrics.

## 8. Evaluation Sequence and Promotion Gates

For each stage:

1. run unit and integration tests;
2. train seed 42 in a new output root;
3. evaluate retrieval, hybrid, generate, and exhaustive modes on validation;
4. run the preregistered validation-only calibration grid;
5. compare with the immediately preceding baseline;
6. promote a seed-42 screening candidate only when validation NDCG@10 is strictly
   higher than the same-seed baseline, no secondary accuracy metric declines by more
   than 5% relative, and there is no correctness or leakage regression;
7. confirm the selected stage with seeds 42/43/44.

A change may be retained for correctness even if its metric effect is neutral, but the
report must distinguish a correctness fix from a demonstrated accuracy improvement.
No claim of improvement is made from a single seed. Final promotion requires higher
three-seed mean validation NDCG@10 than the unchanged baseline. If the paired bootstrap
confidence interval includes zero, report the result as inconclusive rather than
positive, even when the point estimate is higher.

## 9. Testing Requirements

### Unit tests

- structural pools are invariant to extra padding;
- masked attention weights sum to one over valid positions and zero over padding;
- full-model backward reaches every intended branch;
- disabled graph/prototype/SP features reduce to documented fallback paths;
- target-aware scoring is identical between training and inference;
- calibrated fusion preserves seen-item masks and finite outputs;
- training and evaluation identities change only for their declared fields;
- user-balanced sampling is deterministic for a fixed seed and approximately equal by user.

### Integration tests

- tiny end-to-end v4 training produces resumable checkpoints and evaluation reports;
- v3 checkpoints are rejected by v4;
- every inference mode returns legal unseen target items;
- all-zero padding rows remain zero;
- per-user rank output reproduces aggregate metrics.

Tests establish implementation behavior, not recommendation effectiveness. Real-data
validation and multi-seed evidence remain mandatory.

## 10. Expected Files

Initial implementation is expected to touch:

- `cdr_framework/ops.py`;
- `cdr_framework/modules.py`;
- `cdr_framework/text_model.py`;
- `cdr_framework/text_config.py`;
- `cdr_framework/text_training.py`;
- `experiments/run_text_cdr.py`;
- `experiments/sweep_text_fusion.py`;
- focused tests under `tests/`;
- new v4 configuration files and experiment documentation.

Tokenizer files are untouched until P3 is explicitly started.

## 11. Non-Goals

- replacing SIR-CDR with GenCDR or an LLM recommender;
- changing the current data split to obtain a larger headline number;
- using test metrics for model selection;
- silently loading old checkpoints after architecture changes;
- mixing multiple unvalidated changes into one result;
- claiming that architectural intent alone proves a component is effective.

## 12. Rollback and Stopping Rules

Every phase has a separate output root and commit. A phase can be reverted without
altering previous artifacts. Stop expanding a stage when two consecutive controlled
variants fail to improve validation NDCG@10, then diagnose another bottleneck instead
of increasing search breadth. Test evaluation occurs only after the final three-seed
validation result is frozen and documented.
