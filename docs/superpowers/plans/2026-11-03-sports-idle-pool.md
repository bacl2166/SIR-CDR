# Sports Clothing idle GPU expansion

## Design

User authorizes using currently idle GPUs for the existing study. Keep the original
training commit, dataset, SIDs, trial grid, seeds, promotion and NDCG@10 selection.
Add physical GPUs 0--4 to the already extended pool, use at most six workers and
192 GiB host RAM (24 GiB per job). Every dispatch rechecks the original admission
conditions: no compute process, <=5% GPU use, >=12 GiB free VRAM, sufficient RAM.
Existing affinities are immutable. Completed jobs never run again.

The predecessor is the published 47bdc17 GPU7 controller in its own unchanged
checkout. A new versioned resource receipt binds its receipts, current completed
outputs and the new source. Stop only the identity-verified predecessor controller
using pidfd; let its worker finish. Retire that controller, acquire both original
locks, and continue all original stages. This also works if the predecessor has
already exited. Never signal a worker or another user's process. Durable retirement
and recovery receipts make reruns safe; an unknown controller blocks admission.

Alternatives: editing frozen resource JSON breaks provenance; multiple independent
controllers could duplicate jobs; hot editing a running controller changes frozen
source. A separate, audited successor matches the existing deployment workflow.

## Implementation plan

> Execute inline with superpowers:executing-plans; request one final independent review.

- [x] Add failing tests for resource caps, predecessor identity, safe drain/handoff,
  pool dispatch, existing affinity preservation, and completed-job skipping.
- [x] Add experiments/expand_sports_hpo.py. Reuse verified original policy and
  process helpers without changing any existing Python file.
- [x] Run focused tests and full regression suite; review and fix material findings.
- [x] Document fetch, inspect, start and status workflow; commit and push the
  new branch. Deployment remains user-operated: no SSH is available here.

## Verification focus

Controller PID reuse; interruption between TERM and CONT; orphan workers on restart;
predecessor checkout/receipt changes; another controller appearing during handoff;
busy GPUs after an earlier idle sample; completed output and affinity mutation;
wrong job stage; identical promotion/final selection across serial and parallel runs.

## Ledger

2026-11-03: User authorized parallel use of idle GPUs. Snapshot shows 0--4 idle,
5 running c03 reverse, 6/7 occupied by another user. No remote commands executed.

Verification: 8 new tests passed; full suite ran325 tests in28.167s, OK with2
Linux/environment skips on Windows. Independent review found a fresh-ID orphan
worker blind spot; added a red-to-green regression and reject unknown same-study
actors across expansion IDs/checkouts. Transient state/ppid removed from immutable
worker observations. Status can read a running successor, while a second run is
blocked. Resource cap enforced at6 workers and192 GiB. Final independent review
found no remaining material issues, including the user-operated starter script.
Remote deployment remains unexecuted; validate real Linux pidfd on the server.
