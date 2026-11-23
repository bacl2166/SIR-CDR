# Amazon expansion data alignment implementation plan

The user authorized applying the existing Phones–Electronics data processing to Food–Kitchen and Beauty–Electronics.

Use the executing-plans workflow inline. Reuse the current isolated checkout.

## Design

Add a CPU-only preparation entry point. Reuse raw_amazon and build_shared verbatim. Read and verify the existing Phones processed protocol before preparing four new directions. Validate the audited review file hashes and existing Electronics source hashes, download missing metadata on request, then compare every protocol rule and relevant statistics. Record a separate provenance report for the new data. The existing data and model entry points remain the source of processing semantics.

## Tasks

- [x] Write regression tests for real gzip input, all four directions, static-only text, repeated execution and reference/audit drift.
- [x] Implement experiments/prepare_amazon_expansion.py and confirm tests pass against the unchanged shared builder.
- [x] Document the exact server update, download and convert commands, and the remaining embedding/SID stage.
- [x] Check the original preprocessing code diff, commit and publish on the existing candidate-audit branch.

## Validation

Check >=5 valid events in each domain, strict source timestamp, per-domain last-two holdout and all-prefix counts. Assert protocol equality to the verified Phones protocol except dataset identity. Compare raw Electronics hashes to the Phones provenance and all review hashes to the completed audit. Verify counts against audit directions. A metadata-only change must alter processed identity rather than silently reuse data. No model allocation is needed for this stage.

## Review record

An independent review found a gap between the initial review-archive hash check and conversion. Added a failing replacement/recompression test, then bound raw_amazon provenance directly to the completed audit before creating data. The targeted regression suite passed all 20 tests after the fix.

## Continuation after server alignment

The user supplied ALIGNMENT_OK for all four directions. Added prepare_amazon_assets.py to bind the actual Phones embedding revision and tokenizer recipe, automatically admit idle GPUs, run original embedding/tokenize functions in separate child processes, and seal each catalog. Real GPU execution remains a server-side verification step. An independent review identified missing directional manifest binding, weaker checks on ready reuse, the inherited six-GPU pool limit and a stage-boundary occupancy gap. Added regression tests and corrected all four before publication.

Final local verification for asset continuation passed 36 tests. Original dataset conversion, model training and shared asset implementation files have no diff.
