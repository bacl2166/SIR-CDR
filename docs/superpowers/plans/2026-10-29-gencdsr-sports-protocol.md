# GenCDSR-inspired Sports/Clothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox tracking.

**Goal:** Run Sports/Clothing bidirectionally with SIR's unchanged main modules on
a reconstructed, audited GenCDSR-inspired common protocol.

**Architecture:** A pure data converter filters existing Amazon raw events and
delegates paired causal splitting to the existing shared converter with explicit
schema/reference. A separate staged driver reuses existing embedding/tokenizer
and training APIs, seals independent assets, and forbids test before six-model
validation freeze. No edits to model/training implementation.

**Tech Stack:** Python3.10+, pathlib/json/unittest, existing torch/HF/SIR APIs.

**Spec:** docs/superpowers/specs/2026-10-29-gencdsr-sports-protocol.md

## Global constraints

All frozen protocol rules are in the spec. Namespace gencdsr-inspired-sc-v1;
only two directions, seeds42/43/44; no signals; explicit GPU UUID idle gating;
source main and old server jobs unchanged. Public Sports data unavailable means
reconstruction, never exact paper replication. Separate model/asset directories.

Native worktree tool previously cannot resolve this nested repository sourceRoot;
fallback uses ignored .worktrees/gencdsr-sports-protocol, base363ef0b.

## Parallel file ownership and API

Task1 dataset implementer owns cdr_framework/datasets/gencdsr_protocol.py,
tests/test_gencdsr_protocol.py, minimal optional schema/upstream and narrowly
allowed identity-bound protocol_overrides args in
cdr_framework/datasets/gencdr_protocol.py. Exports:
SCHEMA='gencdsr-inspired-sc-v1'; PAIRS tuple of two direction strings;
DOMAINS mapping pair -> (sourceDomain,targetDomain);
convert(root:Path,pair:str)->Path reads raw_amazon and builds namespace;
build_pair(events,items,source,target,output,*,provenance)->Path synthetic interface;
audit(processed:Path)->dict recounts/validates all rows and manifest;
audit_pair_roots(forward:Path,reverse:Path)->dict cross-checks retained events/users.
Data-only module no CUDA imports. Filter audit embedded in protocol provenance.

Task2 driver implementer owns experiments/run_gencdsr_sports.py and
tests/test_gencdsr_sports_runner.py. Imports Task1 API only. Explicit paths for
processed/embedding/tokenizer/control/runtime/model output under SCHEMA. Actions
plan,convert,audit,embedding,tokenize,seal,runtime,smoke,train,validate,freeze,test,
status; pair optional only for plan/convert/audit/freeze/status. Single-pair GPU
action requires --gpu fullUUID. Source root default is this tool checkout,
--data-root points original checkout for raw files; --study-id fresh output ID.
Do not monkeypatch existing module globals. Reuse public APIs directly; copying
small asset glue acceptable, unchanged originals used for all model operations.
Stage validation uses original llm_training.evaluate_checkpoint(...,'validation')
and identity-checked completed manifest; no fusion tuning in first version.
Freeze requires BOTH pairs x ALL3 seeds and binds actual checkpoint/config/source/
asset/validation file hashes; immutable receipts; prohibit any prior test output.
Test is per-pair/per-seed after verified complete freeze and guarded one-time lock.
Caller holds phase locks and GPU admission before torch import; stage subprocesses
recommended to release embedding GPU before tokenizer. CLI can expose each stage
individually, so user commands naturally run separate processes. Do not implement
automatic scheduling; two manually approved independent direction workers may
run concurrently. Runtime configs/ready/freeze use no-overwrite atomic receipts.

Parent owns spec, plan, runbook, integration tests, final review and publication.

## Tasks

- [x] Verify clean baseline CPU suite and source status.
- [x] Dataset TDD: joint fixed-point5/3 pruning within initial shared users (not
  before intersection), cascading deletion, repeat preservation, reverse symmetry,
  causal holdout/graph boundaries, exclusions, metadata fallback, tamper rejection,
  default old shared protocol compatibility. Observe RED before implementation.
- [x] Driver TDD: namespace/source separation, pinned immutable assets/configs,
  physical GPU fail-closed/affinity/locks, no early test, six-model freeze/tamper,
  smoke/train completed/checkpoint guard. Observe RED before implementation.
- [x] Integration with synthetic raw events and CPU tiny catalog/model forward/
  backward verifies unchanged modules and no alternate-data leakage.
- [x] Read-only whole-branch review and fix findings with regressions.
- [x] Full CPU regression, CLI plan/help checks, diff check; record actual results.
- [ ] Commit/push separate feature branch, verify remoteSHA, deliver incremental
  bundle/source runbook. No remote job execution until fresh resource review.

## Evidence 2026-10-29

- Baseline:267 tests/52.409s OK,1 Linux pidfd integration skipped on Windows.
- Final CPU suite:295 tests/36.785s OK,1 skipped (294 passed), local Python3.12 /
  torch2.11.0+cpu. Log delivered outside Git as gencdsr-protocol-cpu-tests.log.
- New protocol28 focused tests GREEN, including original real tinyQwen3+PEFT
  forward/backward with gradients in SIR graph/prototypes/CD/SP injectors/CPF/
  decoder and frozen-base LoRA; no pretrained weights downloaded in this test.
- Whole-branch review found2 P2: freeze did not enforce matching directions,
  and confirmation did not wait for bothpilot42 models. Two regression tests
  reproduced both failures RED; force cross-direction audit under six job locks
  and bind bothpilot42 before non42 training; follow-up staticreview passed.
- Clean detached server deployment supported. GPU admission is stdlib-only until
  physicalUUID/memory/process checks and visibility set; no torch import at CLI
  load. Full model/training files remain byte-unchanged from base363ef0b.
- CLI help/plan and diffcheck passed. Main363ef0b and HPO8644017 remain clean.
- Production raw counts, pruning outcomes, GPU throughput and recommendation
  quality unmeasured here. No server process or existing artifact modified.
