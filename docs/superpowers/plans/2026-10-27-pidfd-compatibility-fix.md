# pidfd compatibility repair

Evidence: server Linux6.12.107, glibc2.41 exports pidfd_open and pidfd_send_signal;
conda-forge Python3.10.22 exposes neither. d857f1c failed at its capability guard,
before any signal. Original controller3120546 remains active. Resource amendment
and intent have already been written and must remain immutable.

Scope: external coordinator only, preserve training checkout a887bd3 and original
18-job protocol/assets/completion rules. Keep current pool4/5/6/7 and cap4.

1. Add regressions reproducing absent Python interfaces and immutable old resource.
2. Add typed libc pidfd adapter with errno handling, no numeric-PID fallback.
3. Probe pidfd on self with signal0 before inspect/run mutations; preserve identity
   checks when signalling the original controller only.
4. Bind only the known legacy failed pre-handoff source to a new immutable repair
   record; keep original resource and intent; bind repair hash to freeze/results.
5. Run focused/full CPU tests, review code, commit and publish to GitHub.
6. Deliver incremental bundle because server cannot reach GitHub. User deploys
   only control checkout, runs real Linux harmless-child test, then read-only
   inspect/GPU status, then handoff in separate steps.

TDD: initial8-test module failed9 subcases due missing backend/binding methods and
the reproduced unavailable Python call. After implementation,7 passed and the real
Linux-only integration test skipped on Windows; original21 controller tests pass.
No local Linux runtime available (WSL absent); do not claim a real server handoff.

Full CPU regression:264 tests in99.112s, OK with1Linux-only skip. Subsequently
added3 focused tests for libc identity reuse rejection, send errno/FD cleanup,
and frozen recovery binding; final focused11 tests OK with1Linux-only skip.
Independent read-only review found no actionable critical/important findings;
actual Linux test and live handoff remain server verification steps.
