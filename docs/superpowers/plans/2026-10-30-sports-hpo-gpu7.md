# Add GPU7 through an auditable successor controller

User authorizes adding GPU7 to the live L20 Sports/Clothing HPO. Existing source8644017, study identity, resources, training data/configurations, candidate budget and per-job affinities are immutable. The controller has no reload interface and refuses orphan workers and altered resource receipts. Direct JSON edits or a second live controller are inappropriate.

## Design

Add a separately versioned coordinator in a new control worktree. It loads and verifies the old tool from the frozen study's original tool root, but never rewrites its files or source identity. Effective resources are the old resources plus explicitly requested GPU7; concurrency2 and RAM budgets remain unchanged. A versioned extension receipt binds old study/resource hashes, new control source, effective resources, completed work, and the handoff.

Safest handoff: validate the exact own old controller using UID, script, arguments, start ticks and pidfd (Python or glibc wrappers); SIGSTOP only that controller to prevent new admissions; let all of its workers finish, verify their original completion receipts/checkpoints; SIGTERM then SIGCONT only the stopped controller after drain, wait for exit, acquire the same original controller.lock and verify no old workers remain. Never signal training workers or rewrite affinities. Save intent and quiesced/drained/retired receipts so interruption can be resumed. A cancel action may resume a still-paused original controller before retirement; never resume after successor dispatch.

New scheduler reuses the original job policy, completion verifiers, resource admission, configurations, promotion and validation/scoring code. It dispatches new internal workers through the extension, whose resources are independently sealed, since old workers correctly reject GPU7. Every worker uses the old job lock, validates extension/study binding and parent identity before importing the original model code. Existing affinity continues to bind old jobs to their original GPUs; only unbound pending jobs can choose GPU7. Done jobs are skipped only after full original validation. Failure stops new admissions and drains other workers. Final validation freeze references both original and extension resource histories, never labels the old whitelist as including GPU7.

## Implementation steps and evidence

- [x] Add test-first pure resource/owner validation and stable-pidfd handoff tests (including Linux-only own subprocess fixture).
- [x] Add resumable handoff and inspect/resume-old/status actions with immutable receipts; preserve frozen files byte-for-byte.
- [x] Add scheduler/worker protocol and lifecycle tests proving no duplicate training, proper GPU7 use and original affinities.
- [x] Full suite317 tests passed (two platform/environment-specific skips on local Windows). Independent review found retirement cancellation and successor recovery issues; both fixed with regression tests, then re-reviewed without important findings. The Linux pidfd fixture must pass on the server before activation.
- [ ] Publish dedicated GitHub branch and provide fixed-revision update/activation commands. No live-server activation has been claimed locally.

No change to NDCG@10 study objective, no test-split evaluation, no changes to native AutoDL dual-NDCG work, no GPU reservation while draining. GPU7 must pass fresh empty-process/VRAM/utilization admission when actually dispatched. The extension cannot guarantee that a card remains available during drain.
