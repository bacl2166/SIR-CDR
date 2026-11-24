# Amazon recommendation continuation implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; perform one independent final code review.

**Goal:** Automatically run all four new Amazon recommendation directions after sealed assets are ready.

**Architecture:** Separate controller plus opt-in policy/evaluator; reuse frozen original data/model/training and snapshot functions.

**Tech Stack:** Python 3.10, PyTorch 2.6, existing Qwen/LoRA environment, GitHub delivery.

**Spec:** docs/superpowers/specs/2026-11-24-amazon-recommendation-design.md

## Constraints and review focus

Preserve original module hashes. Freeze 12 validation bindings before test. Record adapter code in training identity. Missing assets wait; invalid seals fail. GPU/RAM shortages wait; actual failures stop admissions. Physical GPU affinity persists across resume/cache/verify/test. Reject changed configs, assets, policies and test files. Never claim real CUDA validation from CPU mocks.

## Task 1: Opt-in scientific policy and evaluation

- [x] Write failing tests for dual NDCG config and unmodified metric values, signature binding, deterministic fusion selection, cache/verification provenance and immutable test reservations.
- [x] Implement experiments/amazon_recommendation_policy.py with opt-in config, scoped training signature/evaluation adapter and deterministic snapshot evaluation.
- [x] Run policy tests and original checkpoint/recovery regression tests.

## Task 2: Waiting controller and immutable experiment lifecycle

- [x] Write failing tests for missing/changed assets, exact 12-job plan, idle GPU/resource admission, resume affinity, skip completed jobs, sibling failure drain, 12-job freeze and all-seed summary.
- [x] Implement experiments/run_amazon_recommendation.py with run/status and internal workers, separate configs/outputs and frozen validation bindings.
- [x] Run controller tests and existing asset/shared-runner tests; document server worktree/nohup/status commands.
- [x] Obtain independent review, correct findings and rerun appropriate checks: 63 targeted tests passed with no failures/errors.

Delivery: commit and push these tested entry points, then verify local/remote commit equality and provide pinned server startup commands.

## Decisions

The latest request approves configuring the previously proposed automated full pipeline. Use the user's existing dual-NDCG preference and three-seed convention; the default concurrency of two preserves the established shared-server budget. Keep raw HR and NDCG intact and add a separately named mean metric, with an adapter-specific identity.

Independent review found that a replacement controller could exceed the worker limit while orphaned subprocesses kept running. Added a failing regression and a restart gate that waits for owned, same-root, same-batch workers without sending signals. A second regression protects the final non-atomic asset seal by waiting for the existing asset controller lock. Review confirmed both corrections and found no remaining blockers. Integration coverage runs the real controller/queue/freeze/test-receipt/summary lifecycle with only external GPU/model/asset operations mocked; restart performs no additional model work. Separate tiny real CPU-model tests cover the policy, training, exact validation verification and immutable test evaluation. Real CUDA checks remain server-side.
