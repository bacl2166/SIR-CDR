# Qwen latent implementation plan

Goal: deliver the approved Qwen4B+LoRA+two-step persistent-memory combination.
Spec: ../specs/2026-10-07-qwen-latent-design.md
Execution: inline in the existing clean checkout on a feature branch.

- [x] Add failing behavioral tests using an actual tiny random Qwen3 backbone.
- [x] Implement llm_config.py and llm_reasoner.py, including frozen base,
      differentiable feedback, memory reset and strict compact checkpoints.
- [x] Implement llm_training.py and run_llm_cdr.py: prepare pinned model,
      preflight, train/resume/evaluate, accumulation and immutable test report.
- [x] Add three configs and a short server guide with verification limits.
- [x] Run full regression suite and inspect patch (145 tests passed).

Publication: commit the reviewed patch and fast-forward origin/main; report the
verified commit hash in the delivery message.

Review focus: checkpoint omissions; accidentally detached input gradients;
train/eval backbone dropout; final short accumulation batch; relative paths;
mutated model revision; test-report overwrite; frozen base weights saved by
mistake. Existing text-v4 code fingerprints must remain unchanged.

Ruling: microbatch=1 uses cosine alignment. Gradient accumulation does not
create in-batch negatives; reject contrastive_alignment=true at that batch size.
Ruling: structural tokens only in v1, with no raw-history text; the user approved
structural projection, and text conditioning remains a separately evaluated change.
Ruling: branch isolation in the existing clean v4 checkout; do not modify the
older code/ v3 copy in the new desktop workspace.
