# Reverse-direction assets plan

Implement inline following the user's authorization to prepare three reverse tasks in parallel.

Scope: Clothing→Sports, Electronics→Phones, Douban Movie→Book assets only; never start recommendation training or modify forward files. CPU workers run in parallel; optional GPU stage is sequential and refuses a busy GPU.

- [x] New reverse dataset adapter uses unchanged Amazon primitives in reverse domain order. Douban UID mapping, chronological repeats and train-only target tags implemented. Real local reverse split: 215651/1841/1849 rows, 1859 users.
- [x] Protocol/code/raw/processed hashes reject changed or unmanaged outputs. Forward converters/core remain untouched.
- [x] Domain+original-ID+exact-text reuse, frozen source hashes, fast full matrix permutation, pinned coherent partial encoding and resumable chunks implemented. Legacy unknown model commit explicitly recorded, not mixed with new encoding.
- [x] Parallel CPU and serial GPU CLI, task logs/status/locks, optional CPU tokenizer, explicit domain labels and quality gate implemented. GPU phase refuses active forward workflow or ambiguous/busy GPU inventory. No recommender training is launched.
- [x] Ten focused CPU test cases covered split semantics, withheld versus training-positive tags, ID-safe reuse, encoder mismatch, chunk resume/hash checks and scheduling. Independent static review completed; one wording ambiguity clarified (all observed training interactions including training positives are allowed). CLI help/status checked. Publish only reverse files and provide short terminal commands; server GPU execution remains pending.

Reuse of frozen legacy vectors is disclosed as artifact-level reproducibility, not evidence of the original model commit. CPU completion and full embedding/tokenizer completion have distinct status fields. The main command for the currently busy single-GPU server is `--action cpu --workers 3`.
