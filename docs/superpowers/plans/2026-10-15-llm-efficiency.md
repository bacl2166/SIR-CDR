# LLM recommendation efficiency implementation plan

**Goal:** Speed up the 0.6B SIR experiment without dropping samples, changing
metric definitions, reducing reasoning steps, or silently reusing old results.

**Architecture:** Keep Text v4 source and default presets unchanged. Introduce
an LLM-only model subclass with evaluation-scoped item-state caching. Add device
loss accumulation and timing to the isolated LLM trainer. Provide separate fast
presets, a disposable train-only throughput benchmark and a short server runner.

**Authorization:** The user explicitly requested implementation of the previously
discussed design and permits non-equivalent training settings when academically
documented. Execute inline; no further design confirmation is needed.

## Tasks

- [x] First add failing tests for cache equivalence/lifetime and gradient safety,
  weighted detached loss logging, fast presets, and disposable benchmark isolation.
- [x] Implement `llm_model.py`, `llm_efficiency.py`, and integrate in
  `llm_training.py`; sign every new training-code dependency. Cache only fixed
  evaluation states, never across training forwards or parameter updates.
- [x] Add fast presets (batch 4, accumulation 4, eval batch 4, checkpointing off)
  and a train-only benchmark comparing batch/checkpoint combinations at equal
  nominal effective batch. No checkpoint writes or test-label inspection.
- [x] Add a serial prepare/train/validation-tune/verify/test wrapper with unique
  output/cache paths for the first two tasks, refusal of third-task substitution,
  and safe resume of completed tests using existing identity checks.
- [x] Run focused and complete tests; inspect diffs and the full pipeline. Test
  real tiny random Qwen+LoRA locally; report no unmeasured CUDA speed claims.
- [ ] Write server instructions and academic boundaries, then commit and push.

## Acceptance and scientific boundaries

Each prediction retains independent two-step memory, LoRA and CPF. Larger actual
batches change the batch-variance auxiliary statistic and dropout trajectory;
these runs have new config identities and output directories. Old checkpoints
are preserved but not silently resumed under changed code. Keep validation
frequency, scheduler, full data, candidate/seen policy and all metrics unchanged.
Do not cache trainable user representations or detach training graph features.
Benchmark throughput rather than promising a speedup from model size alone.

## Verification ledger

Initial cache/benchmark tests failed on missing implementation, then passed.
One complete regression run passed 168 tests. The new end-to-end fixture and
benchmark/runner suite passed 5 tests, including actual tiny Qwen train, cached
validation tuning, rank verification, immutable final testing and completed reuse.
Independent read-only code review found no actionable correctness issue. Per user
request, do not repeat broad testing without a concrete new failure or concern.
Evaluation time includes the once-per-phase catalog computation.

Server CUDA performance and full recommendation quality are outside these local
CPU regression tests and require the provided server commands. No speedup ratio
or production recommendation score is claimed.
