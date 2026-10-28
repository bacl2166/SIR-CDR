# Sports-Clothing NDCG study plan and implementation contract

User authorizes retraining and tuning both directions, prioritizing NDCG, while
keeping paused Douban/v2 records intact. Implement isolated external controller
using unchanged server training source a887bd389ff72eb14ec96be0dfb876a61dae9331.
Do not incorporate Douban efficiency827407a or change model/data/precision/batch.
Publication: separate GitHub branch and incremental bundle; no SSH available.

## Fixed protocol
- Pairs sports_to_clothing, clothing_to_sports; seeds42,43,44.
- Train8 deterministic configurations per pair on seed42. c00 changes only the
  selection rule to NDCG@10 relative to original defaults (scheduler/early stop
  follow that rule too); c01/02 change LR3e-5/3e-4; c03/04 temperature.05/.2;
  c05 generation_loss_weight.5; c06 cpf_weight.03; c07 combines LR3e-5,
  temperature.05, generation_loss_weight.5, cpf_weight.03. Others remain defaults.
- Rank seed42 screens using native validation N@10/R@10/N@5/R@5, deterministic
  candidate-id tie break; keep2 each pair. Confirm each on seeds43/44.
- Maximum24 NEW training runs, no rerun completed v2 seed42 checkpoints. V2 is
  retained as an external-to-this-study original-code control, not mixed in.
- Collect and bitwise-verify snapshot validation scores for12 finalist models,
  using original recover_llm_evaluation cache+verify (validation only).
- Evaluate fixed48 fusion settings per finalist: pool500/1000, normalization
  none/zscore, weights0,.125,.25,.375,.5,.625,.75,.875,1,1.25,1.5,2.
- Original recovery additionally evaluates17 records (4 settings outside48).
  Disclose65 score records/52 unique settings per model,780 records for12 models,
  excluding native epoch validation and prior exploration. Selection stays48.
- Choose candidate AND one common fusion setting per direction by THREE-SEED
  mean N@10, then R@10/N@5/R@5; report sample std and each seed. Never best seed.
- Record full candidate results, rejected/failed trials, cumulative prior
  observations (v2 native histories, pure retrieval,52 fusion configurations).
- Seal immutable validation-selection freeze with source/data/config/checkpoint/
  cache/verification/selection hashes; NO TEST ACTION in this tool. Untouched
  test files may be fingerprinted by original engine but never scored for HPO.

## Parallel implementation contracts
1. `cdr_framework/ndcg_study.py` pure stdlib policy and
   `tests/test_ndcg_study.py` owned by policy implementer.
   Exports PAIRS, SEEDS, METRICS=(R@5,N@5,R@10,N@10,R@20,N@20),
   candidate_specs()->list[{id,overrides}], fusion_settings()->48settingdicts,
   screen_jobs()->16jobdicts, promote(training_rows)->{pair:[cid,cid]},
   confirmation_jobs(promotions)->8jobdicts, validation_jobs(promotions)->12jobs,
   select_final(validation_rows)->{directions:list,all_summaries:list}.
   Job={pair,candidate,seed}. Training row=job|{metrics:canonical six metrics}.
   Validation row=job|{results:[{setting,metrics},48]}. Validate duplicates,
   coverage, finite/range/example counts; reject invalid/incomplete selection.
   Direction result includes pair,candidate,setting,metrics(mean,std for six),
   seeds:[42,43,44], per_seed:[{seed,metrics}], deterministic ties favor c00,
   smaller pool/weight, none. Accept metrics with canonical R@/N@ aliases.
2. `experiments/tune_sports_clothing.py` and
   `tests/test_sports_clothing_controller.py` owned by controller implementer.
   CLI actions plan/init/status/run/worker. Worker internal only; run requires
   initialized immutable manifest, explicit physicalGPU whitelist, cap<=2.
   Root is unchanged TRAINING_COMMIT a887bd3, external source must be clean and
   separate. Artifacts under root/artifacts/hparam_search/<study-id> only.
   Import original core after sys.path insertion; verify imports stay in root.
   Stable runtime YAML per job; model_config original + overrides + fresh output.
   Worker binds fullGPUUUID, pre-launch idle/free>=12GiB and RAM reservations;
   retains per-job GPU affinity and OS locks. Do not signal any other process.
   Initial orphan worker detection refuses duplicate resume until workers exit;
   per-job lock protects Popen race. Resume original partial checkpoints and
   skip validated completions without rerunning. Failures halt new admissions
   but wait for sibling jobs; no automatic new search budget.
   Train worker invokes original smoke+llm_training.train directly (avoids CLI
   preflight scoring/test paths); validate worker original recover cache+verify,
   then deterministic48-ranking on signed CPU cache. Original raw-score bitwise
   verification implies equal deterministic ranks for all48; bind this derivative
   proof to cache/verification/ranker/selection hashes in immutable receipt.
   Parent validates training manifests against best checkpoint/config/signature,
   validates completed score receipts, promotes once, trains confirmations,
   validates12, selects and freezes all trained-model and scoring provenance.
3. Parent supplies runbook, reviews integration, runs full CPU tests, dispatches
   fresh whole-branch review, publishes isolated branch/bundle and asks user only
   for live GPU inspection/deployment output. Never auto-launch remote training.

## Verification gates
TDD for both modules. Offline fixtures/mock GPU for controller; pure rules tested
exhaustively for24-budget/no test/no best-seed selection and tamper rejection.
No real CUDA data execution local, so no claim of improved test metrics.
User needs fresh GPU snapshot before launch; paused Douban retains its VRAM.

Ruling: budget24 and staged2-finalist method are implementation choices under the
user's existing efficiency/quality delegation; bounded optimum only. Native
worktree tool cannot handle nested repo, so use ignored .worktrees fallback.

## Verification and review evidence (2026-10-28)
- Pure policy17 tests; controller21 and full orchestration2 tests. The latter
  executes a synthetic24-training/12-validation lifecycle, checks complete
  provenance, incomplete-selection rejection and idempotent resume.
- Whole CPU suite:307 tests in60.015s, OK,1 Linux-only pidfd integration skipped
  on Windows. Local runtime Python3.12/torch2.11.0+cpu. No CUDA/model-quality claim.
- Whole-branch review found RAM reservation race, split receipt publication crash
  window and omitted ancillary search counts. Added3 regression tests, observed
  RED, then GREEN. Follow-up review confirms all3 resolved, no important new issue.
- Resource/promotions payload and digest now share one atomic immutable record.
  Available host RAM must cover active reservations plus the new reservation.
- CLI help/plan execute without GPU; source-only additions preserve base main
  and original engine. Server must independently run CPU suite before init and
  inspect/approve current physicalGPU UUIDs before any training is launched.
- Delivery uses a feature branch and an incremental bundle based on363ef0b;
  original server a887bd3 remains untouched. No remote execution performed here.
