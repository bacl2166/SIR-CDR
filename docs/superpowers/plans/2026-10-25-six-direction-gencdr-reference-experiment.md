# SIR-CDR Six-Direction GenCDR Reference Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GitHub-delivered, server-operated workflow that prepares, trains, validates, freezes, tests, and summarizes full SIR-CDR on six cross-domain directions and three seeds without rerunning GenCDR or allowing test feedback into experiment decisions.

**Architecture:** Keep the existing single-pair `gencdr-shared-loo-v1` conversion, asset verification, Qwen3-0.6B training, and snapshot evaluator as the scientific core. Add batch-scoped identities and phase-separated single-job actions, then place a thin six-direction controller above them for read-only server preflight, explicit GPU-whitelist scheduling, validation freeze, one-time test release, and final aggregation. All production commands are run manually by the user after pulling an immutable Git commit; the implementation never opens an SSH connection.

**Tech Stack:** Python 3, PyTorch, Hugging Face Hub, YAML/JSON/CSV, `subprocess`, `concurrent.futures`, Linux `nvidia-smi`, Git, existing `unittest` suite.

**Spec:** `docs/superpowers/specs/2026-10-24-six-direction-gencdr-reference-experiment-design.md`

## Global Constraints

- The six tasks are `sports_to_clothing`, `clothing_to_sports`, `phones_to_electronics`, `electronics_to_phones`, `douban_book_movie`, and `douban_movie_book`.
- Seeds are exactly `42`, `43`, and `44`; all 18 validation jobs must freeze before any batch test job may start.
- The complete model is the pinned Qwen3-0.6B LoRA/two-pass/structural-injection/CPF path already configured by `model_config`.
- Data uses `gencdr-shared-loo-v1`, `all-prefixes`, full-target-catalog ranking, no sampled negatives, HR/Recall and NDCG at 5/10 as primary metrics.
- `paper_alignment_verified` remains `false`; published GenCDR values are descriptive external references only and receive no significance test.
- GPU IDs must be supplied explicitly by the user after manual approval; the controller never uses a GPU outside that whitelist and rechecks a card immediately before every launch.
- The controller never kills another process, changes system CUDA/Python, installs a reverse proxy, or waits indefinitely to seize a newly released GPU.
- All local changes are committed and pushed to `bacl2166/SIR-CDR`; the user manually pulls and runs commands on the server. No SSH password, key, token, raw data, model, checkpoint, or production log enters Git.
- Validation and test are separate CLI phases. The test phase consumes an immutable 18-job freeze manifest and cannot select settings.
- Each `(pair, seed)` has isolated batch-scoped runtime config, output, tuning, log, lock, and status paths; completed identities are never overwritten.
- Code changes follow TDD and retain backward compatibility for existing single-pair commands unless the command would violate the new batch test gate.

## File Structure

- Create `cdr_framework/experiment_control.py`: pure validation, inventory parsing, batch paths, subprocess job records, freeze hashing, and summary helpers; no model imports at module import time.
- Create `experiments/run_gencdr_six.py`: CLI controller for `preflight`, `convert`, `prepare`, `smoke`, `run-validation`, `freeze`, `run-test`, `status`, and `summarize`.
- Create `configs/gencdr_paper_reference.json`: immutable published GenCDR Table 2 values plus citation/scope metadata.
- Create `tests/test_experiment_control.py`: unit tests for GPU inventory, whitelist validation, job state, freeze identity, and aggregation.
- Create `tests/test_gencdr_six_runner.py`: orchestration tests with all subprocess/GPU operations mocked.
- Modify `experiments/run_gencdr_shared.py`: batch-scoped output/runtime paths and phase-separated `train`, `validate`, and `test` actions for one `(pair, seed)`.
- Modify `tests/test_gencdr_shared_runner.py`: identity and phase-boundary tests for the single-job entry point.
- Modify `docs/gencdr_shared_protocol_zh.md`: six-direction manual server workflow and scientific disclosure.
- Modify `README.md`: point users to the new controller and make the validation-before-test order explicit.

## Review Focus

- A whitelisted GPU that becomes busy after preflight must be rejected immediately before launch; Task 2 adds a just-in-time occupancy test.
- A duplicate pair, seed, GPU, malformed batch ID, or out-of-inventory GPU must fail before creating output or launching subprocesses; Tasks 2 and 4 add boundary tests.
- A partial, stale, mixed-commit, mixed-config, or tampered 18-job validation set must not produce a freeze manifest; Task 5 tests every mismatch class.
- Test commands must be impossible before freeze and must remain idempotent only for an identical completed test identity; Tasks 1 and 5 test both cases.
- A failed worker must not erase successful siblings or silently continue to freeze/summarize; Tasks 4 and 6 test nonzero exit isolation and incomplete-batch refusal.

---

### Task 1: Separate one-job training, validation, and test phases

**Files:**
- Create: `cdr_framework/experiment_control.py`
- Modify: `experiments/run_gencdr_shared.py`
- Modify: `tests/test_gencdr_shared_runner.py`

**Interfaces:**
- Consumes: existing `model_config(root, pair, seed)`, `runtime_path(root, pair, seed)`, `verify_ready`, `run_llm_cdr.py`, and `recover_llm_evaluation.py` stages.
- Produces: `validate_batch_id(value: str) -> str` in `cdr_framework.experiment_control`; `model_config(root: Path, pair: str, seed: int, batch_id: str | None = None) -> LLMCDRConfig`; `runtime_path(root: Path, pair: str, seed: int, batch_id: str | None = None) -> Path`; `workflow(..., action: str, ..., batch_id: str | None = None)` supporting `train`, `validate`, and `test` without cross-phase access.

- [ ] **Step 1: Write failing batch-identity tests**

Add `test_batch_scoped_config_isolated_from_legacy_output` asserting that batch `paper-v1` resolves to:

```text
artifacts/llm_cdr_v1/gencdr-shared-loo-v1/paper-v1/<pair>/seed42
configs/runtime/gencdr-shared-loo-v1/paper-v1/<pair>_seed42.yaml
```

Assert the existing no-`batch_id` calls retain their current paths, and reject batch IDs outside `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`.

- [ ] **Step 2: Run the new identity tests and verify failure**

Run: `python -m unittest discover -s tests -p "test_gencdr_shared_runner.py" -v`

Expected: FAIL because `model_config` and `runtime_path` do not accept `batch_id`.

- [ ] **Step 3: Implement batch-scoped path resolution**

Add:

```python
def validate_batch_id(value: str) -> str
def model_config(root, pair, seed, batch_id=None) -> LLMCDRConfig
def runtime_path(root, pair, seed, batch_id=None) -> Path
```

Place `validate_batch_id` in `cdr_framework/experiment_control.py` so later controller code imports one definition. Keep the existing paths when `batch_id is None`; insert the validated batch ID between schema and pair when it is present.

- [ ] **Step 4: Write failing phase-boundary tests**

Add tests using `unittest.mock.patch`:

- `train` calls only `run_llm_cdr.py --action train` and never calls recovery.
- `validate` calls recovery `cache` then `verify`, never `test`, and refuses if `test.json` already exists.
- `test` calls only recovery `test` after ready/config checks.
- legacy `run` remains available for non-batch use, but `--batch-id` with `--action run` is rejected so the batch gate cannot be bypassed.

- [ ] **Step 5: Run the phase tests and verify failure**

Run: `python -m unittest discover -s tests -p "test_gencdr_shared_runner.py" -v`

Expected: FAIL because phase-separated actions do not exist.

- [ ] **Step 6: Implement the phase-separated workflow**

Extend the CLI action choices with `train`, `validate`, and `test`, add `--batch-id`, and route each action through the existing verified runtime config. For `validate`, invoke two fresh deterministic subprocesses in order:

```text
recover_llm_evaluation.py --action cache
recover_llm_evaluation.py --action verify
```

Do not use recovery action `all` in a batch-scoped path. Preserve `run`/`all` behavior only when no batch ID is supplied.

- [ ] **Step 7: Run targeted and regression tests**

Run:

```bash
python -m unittest discover -s tests -p "test_gencdr_shared_runner.py" -v
python -m unittest discover -s tests -p "test_llm_snapshot_recovery.py" -v
```

Expected: all tests PASS; no test report is created by mocked `train` or `validate`.

- [ ] **Step 8: Commit**

```bash
git add cdr_framework/experiment_control.py experiments/run_gencdr_shared.py tests/test_gencdr_shared_runner.py
git commit -m "feat: separate shared validation and test phases"
```

### Task 2: Add read-only server inventory and explicit GPU admission

**Files:**
- Modify: `cdr_framework/experiment_control.py`
- Create: `tests/test_experiment_control.py`

**Interfaces:**
- Consumes: Task 1 `validate_batch_id`, standard library, `torch`, Git CLI, and `nvidia-smi`; all command execution is injectable for tests.
- Produces: `parse_gpu_ids(values: Sequence[str]) -> tuple[int, ...]`; `query_gpu_inventory(run: Callable = subprocess.run) -> list[dict]`; `admit_gpu_whitelist(requested: Sequence[int], inventory: Sequence[dict]) -> dict`; `capture_environment(root: Path, personal_data_root: Path, requested_gpus: Sequence[int], run: Callable = subprocess.run) -> dict`; `assert_gpu_launchable(snapshot: dict, gpu_id: int) -> None`; `write_json_atomic(path: Path, value: object) -> None`.

- [ ] **Step 1: Write failing inventory parsing tests**

Test these exact behaviors:

- CSV rows for two L20 cards parse numeric memory/utilization/temperature/power fields and attach compute PIDs by GPU UUID.
- `parse_gpu_ids` rejects duplicate IDs, negative IDs, non-integers, and an empty whitelist; `admit_gpu_whitelist` rejects an ID absent from inventory.
- malformed or failed `nvidia-smi` raises `RuntimeError` and never returns an “idle” default.
- a GPU with any compute PID fails `assert_gpu_launchable`; an idle whitelisted GPU passes.
- a GPU that was idle in the saved preflight but is busy in a fresh snapshot fails the just-in-time check.

- [ ] **Step 2: Run inventory tests and verify failure**

Run: `python -m unittest discover -s tests -p "test_experiment_control.py" -v`

Expected: FAIL because `experiment_control` does not exist.

- [ ] **Step 3: Implement strict GPU inventory helpers**

Use explicit `nvidia-smi` queries for GPU index/UUID/name/memory/utilization/temperature/power and compute-app GPU UUID/PID/used memory. Treat missing, malformed, or ambiguous fields as refusal, not availability. Do not terminate processes or infer permission from idleness.

- [ ] **Step 4: Write failing environment snapshot tests**

Mock subprocesses and assert the snapshot contains:

```text
schema, captured_at_utc, hostname, user, cwd
git.commit, git.branch, git.status_porcelain, git.remote
python.version, torch.version, torch.cuda_runtime, torch.cuda_available, torch.bfloat16_supported
cpu, load, memory, swap
filesystems for repo, user home, and supplied personal data root
requested_gpu_whitelist, gpu_inventory
admission.allowed_gpus, admission.rejected_gpus, admission.training_started=false
```

Assert a dirty Git tree is reported, not modified, and makes production admission false because its code identity is not reproducible; assert secrets/environment-variable values are not serialized.

- [ ] **Step 5: Implement `capture_environment`**

Use read-only commands and standard-library filesystem calls. Record command failures as explicit diagnostic entries; failure of GPU, Git commit, Python/PyTorch, or filesystem checks makes admission false. Accept `personal_data_root: Path` as an explicit parameter rather than guessing `/Data/<username>`.

- [ ] **Step 6: Run targeted tests**

Run: `python -m unittest discover -s tests -p "test_experiment_control.py" -v`

Expected: all Task 2 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add cdr_framework/experiment_control.py tests/test_experiment_control.py
git commit -m "feat: add audited GPU preflight"
```

### Task 3: Add six-direction asset preparation with safe reverse reuse

**Files:**
- Modify: `cdr_framework/experiment_control.py`
- Create: `experiments/run_gencdr_six.py`
- Create: `tests/test_gencdr_six_runner.py`
- Modify: `tests/test_experiment_control.py`

**Interfaces:**
- Consumes: Task 1 batch paths, existing `run_gencdr_shared.py` `convert`/`embedding`/`tokenize`/`prepare` actions, Task 2 admission helpers.
- Produces: constants `SIX_PAIRS: tuple[str, ...]` and `SEEDS = (42, 43, 44)`; `control_root(root: Path, batch_id: str) -> Path`; `build_asset_jobs(root: Path, pairs: Sequence[str]) -> list[dict]`; `run_parallel_jobs(jobs: Sequence[dict], gpu_ids: Sequence[int], max_workers: int, ...) -> dict`; CLI actions `preflight`, `convert`, and `prepare`.

- [ ] **Step 1: Write failing registry and CLI validation tests**

Assert the registry contains all six directions exactly once. Assert `preflight` requires `--batch-id`, `--gpus`, and `--data-root`; all GPU actions reject `max_workers < 1`, `max_workers > len(gpus)`, more than six GPUs, duplicate GPUs, duplicate pairs, unknown seeds, and malformed batch IDs before launching subprocesses.

- [ ] **Step 2: Run the new runner tests and verify failure**

Run: `python -m unittest discover -s tests -p "test_gencdr_six_runner.py" -v`

Expected: FAIL because `run_gencdr_six.py` does not exist.

- [ ] **Step 3: Implement controller paths and `preflight`**

Write the immutable snapshot to:

```text
artifacts/experiment_control/gencdr-shared-loo-v1/<batch-id>/environment_snapshot.json
```

If the file exists, require identical batch declaration fields while allowing a separately timestamped `latest_inventory.json` for occupancy refresh. Print the allowed/rejected whitelist and a recommended asset/training concurrency; do not launch work.

- [ ] **Step 4: Write failing conversion scheduler tests**

Assert `convert` launches at most three CPU workers, limits `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `TOKENIZERS_PARALLELISM`, uses all six explicit pairs, creates per-pair logs, preserves successful results if a sibling fails, writes a summary with exit codes, and returns nonzero when any pair fails.

- [ ] **Step 5: Implement CPU conversion orchestration**

Launch `run_gencdr_shared.py --action convert --training-examples all-prefixes --pairs <pair>` in batch-independent protocol directories. Use a controller lock and per-pair logs under `logs/gencdr-shared-loo-v1/<batch-id>/`.

- [ ] **Step 6: Write failing GPU asset scheduling tests**

Assert each asset job:

- refreshes GPU inventory immediately before launch;
- binds one physical ID through `CUDA_VISIBLE_DEVICES` and passes logical `--device cuda:0`;
- never exceeds `min(max_workers, len(approved_idle_gpus), 3)` for 8B embedding work;
- never selects an unapproved or newly busy card;
- records PID, GPU, command hash, start/end time, and exit code;
- continues independent siblings after one failure but makes overall preparation incomplete.

Add an exact-text-identity test: reverse-direction embedding reuse is allowed only when `item_texts.jsonl` SHA256, ordered item IDs/domains, pinned model revision/config hash, dimension, normalization, and max sequence length all match; otherwise the reverse direction gets an independent embedding job. Tokenizers are never shared across directions.

- [ ] **Step 7: Implement asset jobs and verified embedding reuse**

Add to `experiment_control.py`:

```python
def embedding_reuse_identity(processed_dir: Path, manifest: dict) -> dict
def can_reuse_embedding(source_identity: dict, target_identity: dict) -> bool
def materialize_verified_embedding(source_dir: Path, target_dir: Path, target_metadata: dict) -> Path
```

Copy atomically rather than hard-linking production tensors, then write a target manifest that records source file SHA256 and target protocol identity. Fall back to normal embedding when any identity field differs. Run direction-specific tokenizer after the embedding is complete.

- [ ] **Step 8: Run Task 2–3 tests**

Run:

```bash
python -m unittest discover -s tests -p "test_experiment_control.py" -v
python -m unittest discover -s tests -p "test_gencdr_six_runner.py" -v
python -m unittest discover -s tests -p "test_gencdr_shared_runner.py" -v
```

Expected: all tests PASS and mocks show no training/test subprocess during `prepare`.

- [ ] **Step 9: Commit**

```bash
git add cdr_framework/experiment_control.py experiments/run_gencdr_six.py tests/test_experiment_control.py tests/test_gencdr_six_runner.py
git commit -m "feat: prepare six shared-protocol directions"
```

### Task 4: Add smoke gating and adaptive 18-job validation scheduling

**Files:**
- Modify: `experiments/run_gencdr_six.py`
- Modify: `cdr_framework/experiment_control.py`
- Modify: `tests/test_gencdr_six_runner.py`
- Modify: `tests/test_experiment_control.py`

**Interfaces:**
- Consumes: prepared/verified assets, approved GPU whitelist, Task 1 `train` and `validate` actions.
- Produces: CLI actions `smoke` and `run-validation`; `JobSpec(pair: str, seed: int, phase: str)`; `launch_job(job: JobSpec, physical_gpu: int, ...) -> JobResult`; `validation_jobs(pairs=SIX_PAIRS, seeds=SEEDS) -> tuple[JobSpec, ...]`.

- [ ] **Step 1: Write failing smoke-gate tests**

Assert `smoke` runs one real training batch per direction on approved GPUs, records peak allocated memory, loss/gradient/ranking checks, and checkpoint writeability, but performs zero optimizer steps and never reads validation/test metrics. A failed direction blocks `run-validation` for the batch.

- [ ] **Step 2: Implement batch smoke orchestration**

Reuse `run_llm_cdr.py --action smoke` after per-pair runtime config preparation. Store one signed smoke report per pair and a batch summary. Compute recommended training concurrency only from approved idle cards and observed peak memory with a recorded safety margin; never alter model dimensions, candidate counts, or data size.

- [ ] **Step 3: Write failing validation-queue tests**

Assert `validation_jobs()` yields exactly 18 unique `(pair, seed)` jobs ordered seed 42 first. Mock two GPUs and assert:

- at most one child per GPU;
- seed 43/44 do not start until all six seed-42 validation jobs succeed;
- a GPU is rechecked immediately before each launch;
- a failed job is recorded and not silently retried with a new identity;
- completed, identity-matching jobs are skipped on rerun;
- incomplete jobs resume through the existing checkpoint path;
- no command contains recovery action `test` or `all`.

- [ ] **Step 4: Implement `run-validation`**

For each job, run Task 1 `train` followed by `validate` in the same assigned worker sequence. Release the GPU after validation verification. Persist scheduler state atomically after every transition (`pending`, `running`, `failed`, `validation-complete`) and include commit/config/data/asset hashes. Stop admission of new jobs when a fresh system snapshot exceeds configured CPU/RAM/disk pressure thresholds, but do not kill running jobs.

- [ ] **Step 5: Test failure isolation and resume**

Run:

```bash
python -m unittest discover -s tests -p "test_gencdr_six_runner.py" -v
python -m unittest discover -s tests -p "test_experiment_control.py" -v
```

Expected: all tests PASS; an injected worker failure leaves successful sibling receipts intact and overall status incomplete.

- [ ] **Step 6: Commit**

```bash
git add experiments/run_gencdr_six.py cdr_framework/experiment_control.py tests/test_gencdr_six_runner.py tests/test_experiment_control.py
git commit -m "feat: schedule audited six-direction validation"
```

### Task 5: Freeze all validation decisions before releasing test

**Files:**
- Modify: `cdr_framework/experiment_control.py`
- Modify: `experiments/run_gencdr_six.py`
- Modify: `tests/test_experiment_control.py`
- Modify: `tests/test_gencdr_six_runner.py`

**Interfaces:**
- Consumes: 18 completed training manifests, best checkpoints, runtime configs, snapshot search reports, and exact validation verification reports.
- Produces: `collect_validation_binding(root: Path, batch_id: str, pair: str, seed: int) -> dict`; `freeze_batch(root: Path, batch_id: str) -> dict`; `verify_freeze(root: Path, batch_id: str) -> dict`; CLI actions `freeze` and `run-test`.

- [ ] **Step 1: Write failing freeze-manifest tests**

Construct fixtures and assert `freeze_batch` refuses:

- 17 rather than 18 jobs;
- duplicate pair/seed entries;
- mixed commit, schema, model revision, protocol recipe, training-example policy, or metric selection;
- incomplete training manifest;
- missing or mismatched exact validation verification;
- an existing `test.json`, `test.lock`, or test reservation before freeze;
- changed checkpoint/config/search/verification bytes after a prior freeze.

For a valid set, assert the manifest records all 18 sorted bindings and SHA256 values, `test_released=true`, `test_results_present=false`, and is byte-stable on identical rerun.

- [ ] **Step 2: Implement immutable validation freeze**

Write exclusively to:

```text
artifacts/experiment_control/gencdr-shared-loo-v1/<batch-id>/validation_freeze.json
```

Create with exclusive semantics or atomic compare; never overwrite a differing file. Include the environment declaration, Git commit, all scientific identities, selected settings, and code/config/artifact hashes.

- [ ] **Step 3: Write failing test-release tests**

Assert `run-test`:

- refuses when freeze is absent or no longer verifies;
- creates exactly the 18 commands `run_gencdr_shared.py --action test --batch-id ...`;
- uses only approved GPUs and the same just-in-time occupancy checks;
- never passes a tuning/search option;
- does not retry a failed immutable test reservation automatically;
- reuses only identical completed test results;
- leaves sibling results intact when one test fails.

- [ ] **Step 4: Implement test release and scheduling**

Use the same one-process-per-GPU queue, but resolve every runtime/checkpoint/setting only through `validation_freeze.json`. Before launching each child, rehash its frozen inputs. A mismatch blocks the whole remaining test queue and writes an audit reason.

- [ ] **Step 5: Run freeze/test gate tests**

Run:

```bash
python -m unittest discover -s tests -p "test_experiment_control.py" -v
python -m unittest discover -s tests -p "test_gencdr_six_runner.py" -v
python -m unittest discover -s tests -p "test_llm_snapshot_recovery.py" -v
```

Expected: all tests PASS; test access is impossible without an intact 18-job freeze.

- [ ] **Step 6: Commit**

```bash
git add cdr_framework/experiment_control.py experiments/run_gencdr_six.py tests/test_experiment_control.py tests/test_gencdr_six_runner.py
git commit -m "feat: freeze validation before final testing"
```

### Task 6: Add scientifically bounded aggregation and GenCDR external reference

**Files:**
- Create: `configs/gencdr_paper_reference.json`
- Modify: `cdr_framework/experiment_control.py`
- Modify: `experiments/run_gencdr_six.py`
- Modify: `tests/test_experiment_control.py`
- Modify: `tests/test_gencdr_six_runner.py`

**Interfaces:**
- Consumes: identical completed 18-job test reports, verified freeze manifest, and immutable reference JSON.
- Produces: `summarize_batch(root: Path, batch_id: str, reference_path: Path) -> dict`; CSV and JSON reports under the batch control directory.

- [ ] **Step 1: Add and test the external-reference fixture**

Create JSON metadata naming the GenCDR paper, Table 2, metric definitions, `strictly_comparable=false`, and these target-domain values:

```text
Sports:     R@5 .0274, N@5 .0261, R@10 .0403, N@10 .0262
Clothing:   R@5 .0181, N@5 .0167, R@10 .0265, N@10 .0203
Phones:     R@5 .0436, N@5 .0411, R@10 .0621, N@10 .0512
Electronics:R@5 .0241, N@5 .0235, R@10 .0342, N@10 .0283
Books:      R@5 .0192, N@5 .0187, R@10 .0237, N@10 .0212
Movies:     R@5 .1713, N@5 .1215, R@10 .1971, N@10 .1275
```

Test the direction-to-target mapping: `clothing_to_sports→Sports`, `sports_to_clothing→Clothing`, `electronics_to_phones→Phones`, `phones_to_electronics→Electronics`, `douban_movie_book→Books`, `douban_book_movie→Movies`.

- [ ] **Step 2: Write failing aggregation tests**

For three known seed values, assert arithmetic mean and sample standard deviation (`statistics.stdev`, denominator `n-1`) for R@5/N@5/R@10/N@10. Assert incomplete, duplicate, nonfinite, mixed-identity, or altered test reports refuse summary. Assert output includes per-seed rows, mean±std rows, absolute/relative descriptive gaps, and the exact non-comparability disclosure; assert it contains no p-value or significance claim.

- [ ] **Step 3: Implement summary generation**

Write atomically:

```text
artifacts/experiment_control/gencdr-shared-loo-v1/<batch-id>/results.json
artifacts/experiment_control/gencdr-shared-loo-v1/<batch-id>/results.csv
```

Bind both reports to the freeze SHA256 and every test SHA256. Preserve metric precision in JSON; format CSV for inspection without changing calculations.

- [ ] **Step 4: Run summary tests**

Run:

```bash
python -m unittest discover -s tests -p "test_experiment_control.py" -v
python -m unittest discover -s tests -p "test_gencdr_six_runner.py" -v
```

Expected: all tests PASS and the disclosure labels GenCDR as an external published reference.

- [ ] **Step 5: Commit**

```bash
git add configs/gencdr_paper_reference.json cdr_framework/experiment_control.py experiments/run_gencdr_six.py tests/test_experiment_control.py tests/test_gencdr_six_runner.py
git commit -m "feat: summarize six-direction reference results"
```

### Task 7: Document the GitHub-to-server operating procedure and verify the whole repository

**Files:**
- Modify: `docs/gencdr_shared_protocol_zh.md`
- Modify: `README.md`
- Modify: `tests/test_gencdr_six_runner.py`
- Track: `docs/superpowers/specs/2026-10-24-six-direction-gencdr-reference-experiment-design.md`
- Track: `docs/superpowers/plans/2026-10-25-six-direction-gencdr-reference-experiment.md`

**Interfaces:**
- Consumes: all prior CLI actions and output paths.
- Produces: complete manual server runbook with no SSH automation.

- [ ] **Step 1: Write a failing documentation contract test**

Assert the Chinese runbook contains, in order, commands for:

```text
git pull --ff-only
preflight --batch-id <id> --gpus <approved IDs> --data-root /Data/<username>
convert
prepare
smoke
run-validation
freeze
run-test
summarize
status
```

Assert the runbook states: human GPU approval is required, tests stay closed until all 18 validation jobs freeze, GenCDR is not rerun, comparison is non-strict, secrets/data/artifacts are not committed, and no SSH connection is provided to the execution agent.

- [ ] **Step 2: Run the documentation test and verify failure**

Run: `python -m unittest discover -s tests -p "test_gencdr_six_runner.py" -v`

Expected: FAIL because the new runbook section is absent.

- [ ] **Step 3: Write the server commands and recovery guide**

Document one command block per phase, using a fixed example batch ID and placeholder approved GPU list. Include `status` checks between phases, log locations, completion criteria, `git rev-parse HEAD`, how to share sanitized status/log output for diagnosis, and the rule that changed code/config requires a new batch namespace.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m unittest discover -s tests -p "test_gencdr_shared_runner.py" -v
python -m unittest discover -s tests -p "test_experiment_control.py" -v
python -m unittest discover -s tests -p "test_gencdr_six_runner.py" -v
python -m unittest discover -s tests -p "test_llm_snapshot_recovery.py" -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run the complete repository suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS with zero failures/errors.

- [ ] **Step 6: Run static CLI checks**

Run:

```bash
python experiments/run_gencdr_shared.py --help
python experiments/run_gencdr_six.py --help
git diff --check
git status --short
```

Expected: both help commands exit 0; diff check is clean; status lists only intended files.

- [ ] **Step 7: Commit design, plan, and runbook**

```bash
git add README.md docs/gencdr_shared_protocol_zh.md tests/test_gencdr_six_runner.py docs/superpowers/specs/2026-10-24-six-direction-gencdr-reference-experiment-design.md docs/superpowers/plans/2026-10-25-six-direction-gencdr-reference-experiment.md
git commit -m "docs: add six-direction experiment runbook"
```

- [ ] **Step 8: Review the complete branch before publication**

Inspect `git log --oneline`, `git diff origin/main...HEAD`, and rerun any test selected by review. Verify no data, checkpoints, logs, credentials, absolute personal paths, or server secrets are tracked.

- [ ] **Step 9: Push the verified commits to GitHub**

Run: `git push origin HEAD:main` only after the user-authorized execution workflow reaches this step and all preceding checks pass.

Expected: push succeeds; record the immutable commit SHA for the server instructions. If `main` rejects non-fast-forward, stop and reconcile without force-pushing.

- [ ] **Step 10: Hand off server execution without SSH**

Provide the exact commit SHA and phase commands. The user manually runs the preflight first, reviews the report, confirms the approved GPU IDs, and only then runs `prepare`/`smoke`. No claim of completed server experiments is made until the user returns signed status/results produced by that commit.

