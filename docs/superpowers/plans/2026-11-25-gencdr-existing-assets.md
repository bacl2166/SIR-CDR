# GenCDR on existing assets implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run a transparent GenCDR reconstruction after the current SIR-CDR Amazon batch finishes, without writing or rebuilding input data.

**Architecture:** Independent `cdr_framework/baselines/gencdr/` model, read-only asset adapter, and `experiments/run_gencdr_baseline.py` scheduler. Published GenCDR tokenizer code is the reference for residual quantization; the unpublished decoder and routers are reconstructed from the paper and explicitly labelled. Frozen Qwen3-0.6B matches the existing SIR backbone; no claim of reproducing Qwen2.5-7B paper scores.

**Tech Stack:** Python 3.10, torch 2.6, transformers 4.53, existing orchestration primitives.

**Spec:** User request: official GenCDR architecture; reuse existing processed assets; run after SIR; complete implementation and GitHub submission.

## Global constraints

- Never invoke raw-data download, preprocessing, PCA, sample filtering, resplitting, or SIR training.
- Default four directions and seeds 42,43,44 exactly match `amazon-expansion-dual-ndcg-disk24-v1`.
- Model-specific tokenizer fitting reads existing embeddings only; writes exclusively baseline output.
- Validate by mean(NDCG@5,NDCG@10), then mean(HR@5,HR@10), then NDCG@10; exact ties earlier epoch.
- Test all seeds only after all validation checkpoints are frozen. Report every seed plus mean/sample standard deviation.
- SIR completion requires verified result/freeze/test receipts, not disappearance of a process.
- Idle GPU means no compute processes, utilization <=5%, >=24 GiB free; bounded CPU/RAM/disk and at most two workers by default.

## Review focus

1. Incomplete/crashed SIR controller never releases the baseline.
2. Changed data or checkpoint invalidates reuse rather than silently training/evaluating different inputs.
3. Padding, duplicate SIDs, seen items and absent beam positives cannot inflate metrics.
4. Training checkpoints exclude frozen pretrained weights but include optimizer/RNG and phase state.
5. Source/target ID mappings remain unchanged; model training never reads test labels.

## Task 1 — read-only assets and completion gate

- [x] Write failing tests for byte-preserving loading, repeat labels, SIR missing/corrupt receipts.
- [x] Implement `assets.py`, `protocol.py`; run these tests.

## Task 2 — reconstructed model and training

- [x] Write failing tiny CPU forward/backward, frozen-universal adaptation and trie ranking tests.
- [x] Implement RQ-VAE, domain low-rank tokenizer/router, universal LoRA experts, domain LoRA and probability router, constrained beam search.
- [x] Implement tokenizer/universal/domain training, resume, validation selection, frozen test evaluation and aggregate reports; execute tiny end-to-end fixture.

## Task 3 — queued deployment and evidence

- [x] Test scheduler admission, completion skipping and immutable declaration/checkpoints.
- [x] Implement automatic wait/resource admission/status, source provenance and Chinese runbook.
- [x] Run affected and regression tests; fresh review; correct material findings.
- [ ] Commit and push dedicated branch; provide pinned server commands. Remote training is not executed locally.

## Decisions

- 2026-11-25: Official repository pinned to `204b0ce0f624d04f84d0c72c52f4d62512ce4bdf`; it does not publish decoder/router training source. Reconstruction must be labelled in every result.
- Existing normalized Qwen embeddings replace upstream OpenAI/PCA features. This is a controlled common-data comparison, not original-paper numerical reproduction.
- Review: retained repeat positives; rejected empty SIR validation provenance and wrong freeze schemas. Added interrupted tokenizer universal/domain and recommendation epoch continuation checks.
- Verification: 364 tests passed with one Linux-only integration skip on Windows; focused baseline suite 11 passed. No CUDA/L20 training has been executed by this local change.
