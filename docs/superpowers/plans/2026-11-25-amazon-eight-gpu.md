# Eight-GPU Amazon scheduler implementation plan

> **For agentic workers:** Use superpowers:executing-plans. Review the complete change before publication.

**Goal:** Fill any idle GPU among indices 0..7 with remaining jobs in the current four-direction, three-seed SIR batch.

**Architecture:** A separate extension imports the exact original clean tool checkout and checks its source against the frozen request. It stops/retires only the original scheduling controller through stable pidfds, adopts live workers without interrupting them, and replaces only the controller queue/resource policy. Original worker code, model configuration, validation freeze, test receipts and result aggregation remain canonical.

**Tech Stack:** Python 3.10, Linux /proc and pidfd/glibc compatibility, existing Amazon runner and OS locks.

**Spec:** User authorizes all eight cards for automatic parallel remaining cross-domain recommendation jobs. GenCDR retains its previous after-SIR gate.

## Constraints and review focus

- No data preparation, model edits, regenerated requests, rerunning complete jobs, or worker signals.
- All eight cards are eligible, but only empty cards with utilization <=5% and smoke-derived free-memory thresholds can receive new jobs.
- Maximum 8 active jobs, including adopted workers; 24 GiB host RAM budget per active job and existing 24 GiB disk gate.
- PID reuse, wrong user/tool/root/batch, ambiguous workers, or changed original source cause refusal before new admissions.
- Persist resource and handoff receipts; restarts discover remaining own workers and skip complete canonical results.
- Complete original validation/freeze/test/summarize pipeline; new GenCDR waiter can observe the same original results.

## Tasks

- [x] Write failing resource/affinity/adoption and pidfd controller-only handoff tests.
- [x] Implement separate `experiments/extend_amazon_recommendation.py` without modifying original runner or its frozen source.
- [x] Verify immediate idle-card dispatch while adopted workers continue, eight-worker cap, failure drain, idempotent completion and PID reuse refusal.
- [ ] Review, run regression checks, write deployment guide and publish GitHub branch.

## Verification and review

- Full regression: 379 tests, OK, 2 Linux-only integrations skipped on Windows.
- Fresh review identified target-process relative path resolution and fork/exec settling risks. Both were corrected and re-reviewed without remaining blockers.
- Added a real Linux pidfd controller/child integration for server-side verification; it has not run on this Windows host.
- Original SIR runner, model, assets and scientific configuration files have no edits. GenCDR CLI resource ceiling now accepts8; default2 unchanged.
