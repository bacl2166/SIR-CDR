# SIR-CDR P0 Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct padding leakage and private-signal asymmetry, separate optimization weights from ranking weights, and add explicit score calibration and evaluation provenance without changing the SIR-CDR architecture.

**Architecture:** Keep the existing semantic item encoder, sparse dual graphs, prototype disentanglement, CD/SP structural injection, implicit reasoner, CPF feedback, semantic-ID decoder, and hybrid ranker. The change repairs data flow around those modules and makes loss/ranking behavior explicit. New checkpoints use identity version `text-cdr-v4`; older checkpoints remain readable only by their original code version.

**Tech Stack:** Python 3.10+, PyTorch, frozen dataclasses, PyYAML, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-20-sir-cdr-performance-improvement-design.md`

## Global Constraints

- Preserve the SIR-CDR module families and their intended roles.
- Do not use the test split for tuning; training and sweeps select on validation NDCG@10.
- Keep padding item id `0` inert throughout sequence and structural aggregation.
- Record all runtime ranking parameters in evaluation reports.
- Store v4 runs under a new output root so v3 checkpoints cannot be overwritten.
- Every production behavior change follows a failing test, then the minimum implementation, then a green suite.

## Review Focus

- Histories with identical real items but different right-padding widths must produce identical CD/SP structural signals.
- Both sequence-private heads and graph-private injectors must affect the fused source/target private signals; disabled SP injection must preserve the sequence-only fallback.
- Training loss weights must affect `forward().total` only; score weights and calibration must affect `rank()` only.
- Calibration must preserve ineligible candidates as `-inf` and remain finite for one eligible candidate.
- Checkpoint identity must reject v3/v4 mixing, while evaluation identity must distinguish runtime scoring settings.

---

## Task 1: Make structural aggregation length-aware

**Files:**

- Modify: `cdr_framework/ops.py`
- Modify: `cdr_framework/modules.py`
- Modify: `cdr_framework/text_model.py`
- Test: `tests/test_text_structural.py`

**Interface:**

```python
def sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor: ...

def sequence_mean(
    embeddings: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor: ...

CrossDomainStructuralInjector.forward(
    ..., source_items, target_items, source_lengths, target_lengths, *, user_shared
)

SpecificDomainStructuralInjector.forward(
    ..., source_items, target_items, source_lengths, target_lengths
)
```

- [ ] Add tests showing that `sequence_mean` ignores padded ids and rejects nonpositive or oversized lengths.
- [ ] Run `python -m unittest tests.test_text_structural` and verify the new assertions fail because lengths are not supported.
- [ ] Implement `sequence_mask` and masked `sequence_mean`; keep the no-length call for unrelated legacy callers.
- [ ] Add injector tests where the same histories are collated to different widths and assert equal outputs.
- [ ] Run the injector tests and verify they fail because injector signatures/attention are not length-aware.
- [ ] Pass lengths through both injectors. Mask CD attention logits before softmax; use masked means in SP injection.
- [ ] Pass `batch.source_lengths` and `batch.target_lengths` from `TextSIRCDR.encode`.
- [ ] Run `python -m unittest tests.test_text_structural` and verify all structural tests pass.
- [ ] Commit with message `fix: mask padded structural histories`.

## Task 2: Fuse sequential and structural private signals symmetrically

**Files:**

- Modify: `cdr_framework/modules.py`
- Modify: `cdr_framework/text_model.py`
- Modify: `tests/test_text_structural.py`

**Interface:**

```python
class GatedSignalFusion(nn.Module):
    def forward(self, sequence_signal: torch.Tensor, structural_signal: torch.Tensor) -> torch.Tensor: ...
```

- [ ] Add a unit test asserting fusion output shape, finite values, and gradients for both inputs and fusion parameters.
- [ ] Run the focused test and verify it fails because `GatedSignalFusion` is absent.
- [ ] Implement a sigmoid gate over the concatenated signals, a convex blend, and layer normalization.
- [ ] Add model tests proving `source_head`, `target_head`, source SP graph input, and target SP graph input all receive gradients in the full variant.
- [ ] Run the tests and verify target/source sequence-private gradients expose the current asymmetry.
- [ ] Add source and target fusion modules. Use fused target-private in the reasoner, query, prefix, and separation loss; use fused source-private in Lsep. When SP injection is disabled, use the sequence-private signal directly.
- [ ] Run `python -m unittest tests.test_text_structural` and verify all structural tests pass.
- [ ] Commit with message `fix: fuse private sequence and graph signals`.

## Task 3: Separate training objectives from inference scoring

**Files:**

- Modify: `cdr_framework/text_config.py`
- Modify: `cdr_framework/text_model.py`
- Modify: `cdr_framework/text_training.py`
- Modify: `tests/test_text_pipeline.py`
- Modify: `tests/test_checkpoint_identity.py`

**Interface:**

```python
retrieval_loss_weight: float = 1.0
generation_loss_weight: float = 1.0
retrieval_score_weight: float = 1.0
generation_score_weight: float = 1.0
fusion_normalization: str = "none"
```

- [ ] Add config tests for nonnegative finite weights, positive training/score totals, and normalization choices `none`, `log_softmax`, `zscore`.
- [ ] Add YAML migration tests: legacy `retrieval_weight`/`generation_weight` map only to score weights; defining both legacy and new score keys raises `ValueError`.
- [ ] Run the config tests and verify they fail on the missing explicit fields/migration.
- [ ] Add the fields and validation to `TextCDRConfig.from_yaml`, removing legacy keys after migration.
- [ ] Add a forward test with zero generation loss weight and a rank test showing loss weights do not change scores.
- [ ] Run the tests and verify the existing total and ranking paths fail those contracts.
- [ ] Update `forward().total` to use loss weights and `rank()` to use score weights.
- [ ] Change `cdr_framework.text_training.VERSION` to `text-cdr-v4`; update identity tests so v3 checkpoints are rejected.
- [ ] Run `python -m unittest tests.test_text_pipeline tests.test_checkpoint_identity` and verify all pass.
- [ ] Commit with message `feat: separate loss and ranking weights`.

## Task 4: Add masked score calibration

**Files:**

- Modify: `cdr_framework/ops.py`
- Modify: `cdr_framework/text_model.py`
- Modify: `tests/test_text_structural.py`

**Interface:**

```python
def calibrate_scores(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    mode: str,
    eps: float = 1e-6,
) -> torch.Tensor: ...
```

- [ ] Add direct tests for `none`, `log_softmax`, and `zscore`, including masked rows with one eligible candidate.
- [ ] Run the focused tests and verify failure because the helper does not exist.
- [ ] Implement calibration per row. Apply statistics only to eligible values and restore all ineligible positions to `-inf`.
- [ ] Add a hybrid-ranking test showing changing `fusion_normalization` changes ranking input scores without unmasking seen items.
- [ ] Run it and verify failure against the current dynamic `fusion_norm` path.
- [ ] Calibrate retrieval and generation candidate scores before weighted fusion, using the explicit config field.
- [ ] Run `python -m unittest tests.test_text_structural` and verify all pass.
- [ ] Commit with message `feat: calibrate hybrid ranking scores`.

## Task 5: Record evaluation identity and update the sweep

**Files:**

- Modify: `cdr_framework/text_training.py`
- Modify: `experiments/sweep_text_fusion.py`
- Modify: `experiments/run_text_cdr.py`
- Modify: `tests/test_checkpoint_identity.py`
- Modify: `tests/test_text_fusion_sweep.py`

**Interface:**

```python
def evaluation_identity(config, checkpoint_path, split, mode) -> dict[str, object]: ...
```

- [ ] Add tests asserting the identity includes checkpoint hash, split, mode, rerank count, temperature, both score weights, normalization, and batch size.
- [ ] Add a test asserting two runtime scoring configurations produce different evaluation identities without changing checkpoint bytes.
- [ ] Run both test modules and verify failure on missing fields/dynamic attribute handling.
- [ ] Implement `evaluation_identity` and include it in individual evaluation JSON.
- [ ] Replace dynamic `fusion_norm` injection in the sweep with dataclass replacement. Rename report fields to score weights and accept all three normalization modes.
- [ ] Update the CLI description from v2 to v4.
- [ ] Run `python -m unittest tests.test_checkpoint_identity tests.test_text_fusion_sweep` and verify all pass.
- [ ] Commit with message `feat: track v4 evaluation identity`.

## Task 6: Add a safe v4 configuration and operating guide

**Files:**

- Create: `configs/text_sports_clothing_v4_p0.yaml`
- Modify: `README.md`
- Test: `tests/test_experiment_config.py`

- [ ] Add a config-loading test asserting the v4 output root, full architecture flags, separate weights, and explicit calibration.
- [ ] Run it and verify failure because the v4 config is absent.
- [ ] Create the config with output root `artifacts/text_cdr_v4/qwen3_8b/sports_to_clothing`, selection metric `NDCG@10`, loss weights `1.0`, score weights `1.0`, and `fusion_normalization: log_softmax`.
- [ ] Document commands for training, validation-only fusion sweeps, and one final test evaluation. State that the v3 checkpoint cannot be evaluated by v4 code and must not be copied into the v4 output root.
- [ ] Run `python -m unittest tests.test_experiment_config` and verify all pass.
- [ ] Commit with message `docs: add SIR-CDR v4 run configuration`.

## Task 7: Whole-repository verification and push preparation

**Files:**

- Verify all tracked changes and generated reports.

- [ ] Run `python -m unittest discover -s tests -v` and save the complete result in the execution ledger.
- [ ] Run `python -m compileall cdr_framework experiments tests`.
- [ ] Run `git diff --check` and `rg -n "fusion_norm|retrieval_weight|generation_weight" cdr_framework experiments tests configs README.md` to confirm remaining legacy names are intentional migration tests or old reproducibility configs.
- [ ] Review `git diff origin/main...HEAD` against the spec and the Review Focus cases.
- [ ] Commit any verification-driven corrections with focused messages.
- [ ] Push the tested branch to `origin/main`.

