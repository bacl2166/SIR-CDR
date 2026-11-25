# Lower Amazon recommendation disk gate

User authorized lowering the disk threshold and deploying. The server reported 29.242 GiB free, four ready assets verified and zero training jobs. Use a 24 GiB startup reserve and a new batch ID, amazon-expansion-dual-ndcg-disk24-v1, since the old immutable request binds the old controller source. Keep the old declarations/logs and reuse all prepared assets without conversion or SID training.

- [x] Add failing tests for the observed capacity, the exact 24 GiB boundary, insufficient-space wait and the new declared resource policy/batch identity.
- [x] Replace the 32 GiB abort with an honest capacity check and wait below 24 GiB. Persist the new minimum in the request. Update documentation/default batch.
- [x] Run relevant regressions: 66 tests passed with no failures/errors. Independent review found no blockers. Original asset/model/evaluation files are unchanged.

Delivery: commit/push the tested patch; provide pinned server update/nohup/status commands and an offline Git bundle fallback. No SSH connection is available, so actual deployment is performed through the supplied server commands.

The reserve is operational, not a measured scientific minimum; training parameters, seeds, dual-NDCG selection, cache identities and test freeze are unchanged.
