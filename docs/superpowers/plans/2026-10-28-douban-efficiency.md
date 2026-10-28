# Douban efficiency implementation plan

> Execute inline using superpowers:executing-plans; final independent review.

**Goal:** Reduce train-only Python storage/collation and redundant sequence-length
transfers, without dropping prefixes, changing objectives or updating the live run.
**Architecture:** Preserve evaluation loading/masks. Train-only rows omit unused
full seen sets, collate only tensors, and keep sequence lengths on CPU for packing.
Extend existing disposable benchmark with explicit legacy/optimized input paths,
uniform seeded train-only samples, idle GPU admission and reproducible reports.
**Stack:** Python3.10+, torch2.6, existing LLM model and immutable identities.

**Authorization:** User requested efficiency implementation after approving the
isolated approach. Use independent worktree and GitHub delivery. Active server
a887bd3/paper-reference-v2 and control363ef0b must remain unchanged. Native worktree
tool rejected nested repository sourceRoot, so use ignored .worktrees fallback.

**Alternatives:** no prefix subsampling (changes objective); no default batch
increase (CPF variance depends on microbatch); no trainable graph cache (stale
values/gradients). Default8x2 remains;16x1 is benchmark-only pending quality tests.

## Tasks
- [x] RED: test omitted seen sets preserve all row tensors/validation masks and
  malformed input checks; test collation does not iterate seen data.
- [x] Implement training-only loader/collator and CPU-length batch transfer in
  signed existing files, keep shared runner/config/assets implementation unchanged.
- [x] Compare all losses and parameter gradients against legacy on identical
  weights, RNG and rows. Test original train/resume/evaluation suites.
- [x] Extend train-only benchmark with recorded pipeline and uniform row order;
  test reproducible samples, no checkpoint writes/RNG mutation, reject busy GPU.
- [x] Run full CPU suite and optional CUDA equality test (skip if unavailable).
- [x] Fresh whole-branch review, document server isolation and benchmark commands,
  commit/push GitHub branch and deliver incremental bundle. Never auto-adopt16x1.

## Verification boundaries
CPU tests establish formula/data/gradient equivalence for unchanged8x2 on fixtures.
GPU numerical agreement, throughput/RSS and validation quality require server
measurements on an idle approved card. No promised speedup or bitwise CUDA claim.
Use a new batch/output: source signatures intentionally reject old checkpoints.

## Execution evidence
- Baseline267 tests passed53.755s,1Linux skip.
- RED initially failed import of missing training_batch_to_device. Further GPU
  utilization/memory rejection regression failed before adding resource checks.
- Final277 tests passed52.641s,2skips (Linux pidfd integration and CUDA equality).
- CPU complete optimizer trajectory with dropout matches legacy tensor-for-tensor;
  loss/gradient, full evaluation history masking and both preset parity tests pass.
- CLI help and git diff --check pass. Independent whole-branch reviewer found no
  important issues including source, tests, presets, benchmark guard and runbook.
- Ruling: preserve default8x2 and existingHR@10 selection to isolate efficiency;
  do not implement sampling, bigger default batch, multiprocessing, precision
  changes or interruption/migration of the active checkpoint.
- Ruling: no CUDA speed/RSS or actual server deployment was performed locally.
  Publication is an isolated feature branch, leaving main363ef0b unchanged.
