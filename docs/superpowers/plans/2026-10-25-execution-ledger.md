# Execution ledger — six-direction reference experiment

Plan: 2026-10-25-six-direction-gencdr-reference-experiment.md
Base: b49e35ea945ba1f094002a6f21e31db5ae1d06b9

- Tasks 1–2 core complete: batch paths and phase dispatch, fail-closed GPU inventory, read-only host snapshot. 10 focused tests passed; commit 76cd683.
- Ruling: use an ordinary local feature branch in the existing clean checkout — user authorized efficient implementation and GitHub delivery; avoids duplicating the checkout while isolating main.
- Ruling: preserve existing single-task/global-idle workflows; new batch workers use per-card admission owned by the controller — existing global guard cannot support parallel jobs.
- Ruling: unit-test temporary fixtures require unsandboxed execution on this Windows host — default sandbox temporary directories reject writes; no production GPU work is performed.
- Ruling: human-readable runbook prose receives manual review, not exact-text tests — behavior tests cover CLI gates, avoiding tests that only duplicate documentation wording.
- Preflight interfaces: Tasks 1→3/4 share batch paths; Tasks 2→3/4 share whitelist admission; Tasks 4→5 share validation receipts; Tasks 5→6 share frozen file hashes. All later tasks must consume these exact records.

- Tasks 3–7 implemented in commit 23ae641: six-direction assets/queue, 18-job freeze, indirect reference summary, manual server runbook. Initial full suite: 219 tests passed.
- Final independent review by gpt-6-astra found 3 P1 and 4 P2 issues; all entered one RED→GREEN fix pass.
- Final fixed: busy-card recheck dropping pending jobs — `test_last_moment_busy_card_cannot_drop_job` and `test_missing_returned_jobs_block_later_seeds` reproduced failures before fix.
- Final fixed: physical-index/CUDA enumeration mismatch — `test_child_cuda_visibility_uses_full_uuid` and `test_child_rejects_wrong_logical_gpu_before_model_allocation` reproduced failures before UUID binding and child verification.
- Final fixed: fictitious freeze and incomplete global validation — `test_fictitious_eighteen_job_freeze_is_rejected` reproduced acceptance before canonical reconstruction of all18 bindings.
- Final fixed: freeze failure allowing subsequent test admissions — `test_fatal_admission_stops_pending_jobs` reproduced missing fatal gate before queue admission change.
- Final fixed: completed work requiring an affinity GPU — `test_completed_jobs_skip_gpu_admission_entirely` reproduced missing CPU reuse before implementation.
- Final fixed: no concurrent host/cache/output budget — `test_concurrent_workers_reserve_host_memory`, `test_storage_check_includes_actual_cache_filesystem`, `test_asset_ram_budget_uses_pilot_peak_with_margin` failed before reservations, actual-filesystem budgeting and first-task telemetry.
- Final fixed: edited completed metrics accepted before first summary — `test_metric_edit_before_first_summary_is_rejected` failed before immutable completion receipts.
- Final: Ruling: elevate checkpoint-write probe and pinned existing-embedding validation from review minors to required fixes — both are explicit experiment prerequisites; wrong decisions could admit unwritable checkpoints or mixed encoder assets. Their regression tests failed before implementation.
- Final: Ruling: driver/BF16/raw-asset snapshot gaps are required preflight work — added read-only capability and asset inventory; BF16 queried through nvidia-smi rather than allocating a controller context on an unapproved GPU.
- Final: Ruling: canonical global freeze validation rehashes all18 jobs before each final-test admission — higher I/O overhead is accepted to preserve the approved whole-batch freeze rule.
- Final: minor (deferred): child PIDs and exact commands remain in persistent phase logs rather than duplicated in status JSON; status shows queue state, GPU, time, error and result. Actual server GPU peak/numerical behavior remains a required user-run smoke/validation check.

- Tasks 1–7: complete. Final whole-repository verification after the fix pass: `python -m unittest discover -s tests -v` → 232 tests, 0 failures/errors, 35.012s. CLI help and `git diff --check` passed.
- Final: Ruling: apply the user's existing GitHub-delivery authorization as a fast-forward update to main — no extra approval menu; remote main was confirmed unchanged at the implementation base. A concurrent remote update must stop publication rather than trigger force-push.
- No production training, server login, or remote GPU inspection was performed locally. The runbook starts with user-operated server hardware inspection and manual GPU approval.

## 2026-10-26 — fresh CUDA worker smoke failure

- Server evidence: all six asset preparations completed, then all six fresh smoke subprocesses failed at `torch.cuda.reset_peak_memory_stats(cuda:0)` before model construction under PyTorch 2.6.0+cu124. The parent worker's UUID verification does not initialize the separate `run_llm_cdr.py` process.
- Root cause checked against PyTorch v2.6.0: peak-stat reset enters the allocator without lazy initialization; CUDA `set_device` performs device lazy initialization.
- RED: `test_cuda_smoke_initializes_selected_device_before_resetting_peak_stats` failed with the same `Invalid device argument` on the original implementation. The CPU-bypass regression passed.
- Fix: select/initialize the requested worker device before resetting the peak counter. Resolve bare `cuda` through its current logical device, preserving the CLI default; its regression also completed RED/GREEN. No measurement is skipped; model loading, forward/backward, and ranking remain inside the measured interval.
- GREEN: final focused CLI suite 6/6 passed. Final full CPU suite 235/235 passed in 38.490s, using local PyTorch 2.11.0+cpu. Actual PyTorch 2.6 CUDA execution still requires the user's server smoke checks. Read-only review of the supplied diff found no actionable issues; review tools could not launch, so the review's independent filesystem/GPU evidence is limited.
- Recovery: keep `paper-reference-v1` and failed receipts unchanged; use a new batch such as `paper-reference-v2` after pulling the fix. Shared converter, asset driver, protocols and sealed data remain unchanged, so `prepare` can verify/reuse all six ready assets and write new batch runtime configs. No formal training or final testing had started in the failed batch.
- Resource ruling: retain the approved GPU whitelist; maximum concurrency remains three, actual admission uses currently idle cards only. The latest user-provided process list shows GPU4/5 occupied; do not stop or reuse these processes to recover this experiment.
