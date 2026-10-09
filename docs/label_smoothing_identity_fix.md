# Label smoothing training identity correction

Text v4 previously removed `label_smoothing` from the shared configuration
payload. Runs differing only in this training objective could therefore share
an identity and reuse a completed manifest or resume an incompatible checkpoint.
The old test explicitly expected this omission; it has been replaced by
regression tests for identity, reuse, resume, and evaluation checks.

The configuration payload now retains the field, including its zero value.
This does not change the loss implementation or model architecture.
Existing exact configuration comparisons reject both a different value and a
missing legacy field. `--allow-code-mismatch` only relaxes code fingerprints;
it cannot bypass this configuration mismatch.

## Existing experiments

- Preserve previous checkpoints, manifests, configurations and logs for audit.
- Old Text v4 identities lacking this field are not accepted as verified runs,
  even when the requested value is zero. Do not add a guessed value to old
  identities or copy a base checkpoint into an L3A output directory.
- Retrain recommendation models in a **new output root** with the corrected
  code, including both the base and label-smoothed comparison where applicable.
  Existing offline embedding/tokenizer assets can be reused if unchanged.
- Previously reported metrics are historical results with an incomplete
  objective identity. The bug establishes a possibility of incorrect reuse,
  not proof that a particular historical run was reused. Audit those runs
  using their original configuration and logs before drawing conclusions.
- No automatic compatibility exemption is provided. LLM v1 already retained
  this field in its independent configuration payload; its strict code hashes
  may still detect this shared-file change.

Verification: `python -m unittest discover -s tests -p test_checkpoint_identity.py`.
