# Elastic Six Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the existing live batch to six direction queues without interrupting workers or repeating completed seeds.
**Architecture:** A separately versioned standalone coordinator imports unchanged pinned training modules. It adopts own-user workers, records an append-only resource amendment, and binds that amendment to the canonical18-job final freeze.
**Tech Stack:** Python3.10 stdlib orchestration, Linux /proc and pidfd; existing PyTorch/model code unchanged.
**Spec:** docs/superpowers/specs/2026-10-27-elastic-six-handoff-design.md

## Global Constraints

- Original checkout commit, manifest, training signatures and existing artifacts remain unchanged.
- User-approved GPU0/1/4/5/6/7 only; max6; prefer uncovered directions, fill spare slots with independent seeds42/43/44, never repeat a job.
- Never signal training/validation workers, process groups, or another user.
- All18 exact validation bindings required before any final test.
- Code is delivered through GitHub, then a separate server control worktree; no SSH.

## Review Focus

- Controller PID reuse/wrong owner: stable pidfd plus full identity verification.
- Fork/exec overlap at handoff: rescan own batch processes after controller exit before scheduling.
- Restart after coordinator death: live-worker adoption and immutable amendment/provenance verification.
- A complete checkpoint with missing validation: skip training and finish only validation.
- Unavailable affinity card or final-test failure: preserve receipts, stop admissions, no forced migration or retest.

### Task 1: Standalone process and provenance boundary

Files: create experiments/extend_gencdr_run.py and tests/test_extend_gencdr_run.py.
Interfaces: process snapshots with PID/start tick/UID/cwd/argv/expected GPU UUID; append-only receipt verification; pinned original backend loader.
- [ ] Write and run failing tests for wrong PID/action/UID, changed receipts, and batch/source boundaries.
- [ ] Implement stdlib-only process/provenance helpers, inspect and pidfd handoff.
- [ ] Verify focused tests pass.

### Task 2: Adopted six-direction rolling queue

Files: same executable and tests.
Interfaces: next eligible jobs from canonical completed set, preferring uncovered directions before filling with independent seeds; adopted processes count toward slots; stage reuse from original checkpoint and validation validators.
- [ ] Write and run failing tests for no completed work rerun, direction priority and independent seed filling, six slots, adoption, and occupied cards.
- [ ] Implement rolling queue, preserved UUID affinity, resource gates, incomplete-stage continuation.
- [ ] Verify focused tests pass.

### Task 3: Freeze, recovery, and delivery

Files: executable/tests; create docs/elastic_six_server_runbook_zh.md.
- [ ] Test original18-job freeze plus supplemental resource freeze verification before test admission and final provenance.
- [ ] Implement stop-on-failure and resumable worker adoption; document exact server inspect/run commands.
- [ ] Run full CPU suite and CLI help, inspect diff for unchanged training files.
- [ ] Obtain read-only code review, fix actionable issues, commit/push, give concrete server commands.
