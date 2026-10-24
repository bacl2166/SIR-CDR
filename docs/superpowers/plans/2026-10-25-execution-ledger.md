# Execution ledger — six-direction reference experiment

Plan: 2026-10-25-six-direction-gencdr-reference-experiment.md
Base: b49e35ea945ba1f094002a6f21e31db5ae1d06b9

- Tasks 1–2 core complete: batch paths and phase dispatch, fail-closed GPU inventory, read-only host snapshot. 10 focused tests passed; commit 76cd683.
- Ruling: use an ordinary local feature branch in the existing clean checkout — user authorized efficient implementation and GitHub delivery; avoids duplicating the checkout while isolating main.
- Ruling: preserve existing single-task/global-idle workflows; new batch workers use per-card admission owned by the controller — existing global guard cannot support parallel jobs.
- Ruling: unit-test temporary fixtures require unsandboxed execution on this Windows host — default sandbox temporary directories reject writes; no production GPU work is performed.
- Ruling: human-readable runbook prose receives manual review, not exact-text tests — behavior tests cover CLI gates, avoiding tests that only duplicate documentation wording.
- Preflight interfaces: Tasks 1→3/4 share batch paths; Tasks 2→3/4 share whitelist admission; Tasks 4→5 share validation receipts; Tasks 5→6 share frozen file hashes. All later tasks must consume these exact records.
