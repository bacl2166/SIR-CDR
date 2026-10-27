# Existing-batch six-direction resource expansion

User authorization: use remaining idle cards for six concurrent recommendation jobs; never redo a completed direction/seed, but later seeds may run. Earlier authorization permits efficient implementation and GitHub delivery. No SSH access. Preserve current training and validation work.

## Chosen approach

Deliver a standalone coordinator in a separate Git worktree on the server. Keep the active training checkout at its original commit, so model/checkpoint identities, ready assets, smoke gates and canonical freeze remain valid. A new immutable resource amendment records the original declaration hash, coordinator code provenance, six approved physical GPU UUIDs, and direction-priority scheduling of independent fixed seeds. This explicitly supplements the initial GPU declaration without editing it.

The pool is GPU0/1/4/5/6/7, retaining every initially approved card. GPU2/3 are unused reserves. Maximum concurrency6, prefer uncovered directions and lower seed numbers. A completed42 permits that direction's43 immediately; other directions need not finish42 first. If fewer than six distinct directions still have work, independent seeds of the remaining directions can fill otherwise idle slots. Never repeat a direction/seed. All18 validated jobs must still freeze before any final test.

Alternatives rejected: a second ordinary controller conflicts with the existing lock and pending queue; changing the live checkout invalidates the batch commit; a new batch/checkpoint rewrite needlessly discards provenance and progress.

## Handoff and adoption

Only the verified own-user original run-validation controller may receive SIGTERM, via a Linux pidfd after UID, start tick, exact script/root/batch/action verification. Do not SIGSTOP it: that would leave a stopped controller if the helper crashed. Never signal a process group or a training/validation worker. Write durable handoff intent before signalling. The old outer set-e shell exits and cannot advance to freeze/test. Existing children continue running. Acquire the released original controller lock, rescan matching own-user worker processes (including inner processes identified by the exact runtime config), and adopt their GPU UUIDs and process start identities. Repeated scans cover a last-moment worker launch. A coordinator restart reconstructs adoption from live processes and immutable receipts, including orphan inner training after a wrapper exit.

For each adopted job, wait until its worker processes finish, recheck GPU admission, and continue only incomplete stages. A verified training manifest skips training; a canonical exact-validation binding skips the whole job. Completed work is never reset or overwritten. Preserve affinity for training/validation/test. Fail on ambiguous workers, foreign GPU occupants, missing affinities, changed sources/assets, or any prior test/freeze when starting a new amendment.

## Scheduling and testing

Reserve existing adopted jobs in the six slots and RAM budget before new admissions. Every15 seconds, check only the approved pool; recheck process occupancy and memory immediately before each launch. Six direction queues allow later seeds to fill idle slots. Resource pressure blocks admissions; failure stops new admissions while existing jobs finish.

Use the original immutable freeze and test-completion implementation from the pinned training checkout. A supplementary immutable freeze binds the resource amendment and coordinator provenance to the canonical18-job freeze; verify both before every test admission. Bind final results hashes into a supplementary provenance receipt. Do not relax any checkpoint, asset, runtime or evaluation identity check.

Only initial takeover rejects an already-frozen/test-accessed batch. A restart with an existing verified amendment may continue its own freeze/testing; a crash between canonical freeze creation and supplemental receipt creation is recovered by verifying the canonical freeze and sealing the deterministic supplemental binding before any test admission.

## Delivery and verification

New executable and tests only; do not modify existing training, data, model, validation, smoke, driver or orchestration sources. GitHub commit -> server git fetch -> separate detached control worktree -> inspect -> handoff/run. Test process identity and pidfd signalling with fakes, scheduling/dependencies, non-retraining reuse, duplicate ownership, immutable receipts, all18 release, and resumption. Real Linux/GPU handoff remains a user-operated server verification boundary.
