# Amazon recommendation continuation

The user requested automatically starting complete bidirectional recommendation experiments after the four new asset catalogs finish. This implements the already discussed training → validation selection → freeze → test sequence, preserving automatic idle GPU discovery and the existing data assets.

Approach: a separate controller and opt-in policy/evaluation adapter reuse the unchanged model factory, training loop, deterministic snapshot scoring and ready validation. Extending the six-direction controller would impose its fixed 18-job freeze; changing the asset controller would change the source hash bound to existing receipts. Independent entry points avoid both conflicts.

Run Food↔Kitchen and Beauty↔Electronics, seeds 42/43/44, in a new batch directory. Use the existing Phones model/training defaults, except checkpoint selection uses the user's previously chosen equal mean of NDCG@5 and @10. Exact mean ties retain the earlier epoch during training. Fusion selection uses that mean, then HR@5/@10 mean, NDCG@10 and a deterministic setting tie break. All attempted settings and all four metrics are retained. No promise of improvement over GenCDR is made.

Wait until all four ready seals exist and validate them before GPU work. Check cached pinned Qwen weights, CPU/RAM/storage, and run zero-update smoke tests. Requery GPUs every admission: no compute processes, utilization <=5%, at least 12 GiB free or the measured smoke requirement if greater. Default two workers; temporary resource shortages wait 60 seconds. Stop admissions on real child failures and drain siblings. No modification of existing original assets, model modules or old experiment outputs.

Keep each seed's training, validation cache, exact verification and test on the same GPU UUID because the existing snapshot identity binds physical hardware. Whole-batch validation freeze must bind all 12 checkpoints/configs/assets/cache/search/verification files before any test. Completed artifacts are skipped only after identity/hash checks. Failed or unfinished test reservations are never silently overwritten. Final results include every seed and per-direction sample mean/std for HR/NDCG @5/@10/@20.

Commands are delivered through a pinned GitHub commit into a separate server worktree, leaving the running asset worktree alone. The new background controller can start waiting immediately; it does not start or duplicate the asset preparation controller.
