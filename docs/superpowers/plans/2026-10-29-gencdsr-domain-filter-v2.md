# Domain filtering scope correction

1. RED: external/non-common users support globally valid items; filtering their
   support away must not create a common-only item core. Independent domain
   pruning must not remove opposite-domain events of a now non-common user.
2. GREEN: independent full-domain fixed points then original-user intersection;
   version schema/study; audit full-domain and overlap trace separately, optional
   independent raw replay. Preserve every SIR model/training source file.
3. Update old tests to revised declared scope; add tamper/raw-replay/namespace
   regressions. Keep causal, cross-direction and real tinyQwen gradient tests.
4. Run focused and full CPU suites and CLI; independent source review, fix all
   important findings; inspect final diff/model-source identity.
5. Commit/push isolated v2 branch, verify remoteSHA, deliver exact online server
   worktree + CPU/data-only commands. Preserve v1 asset and diagnose counts first.

No GPU training or recommendation-quality claim before server audit and runs.

## Local evidence

- Two regression failures reproduced against original published dfafeff9 filter
  by loading that exact Git blob, not by editing its pinned server checkout.
- v2 focused30 tests/2.274s OK, including tinyQwen/LoRA SIR gradient integration.
- Full CPU suite297 tests/25.984s OK,1 Linux-only test skipped on Windows.
- Independent read-only review: no important findings; reviewer separately ran
  protocol12 tests/0.804s OK and diffcheck passed.
- CLI plan reports v2 schema/study and full-domain scope. Diff confirms original
  model, training, reasoner and tokenizer-training sources untouched.
- Actual full-domain raw replay/counts await new independent server deployment.
