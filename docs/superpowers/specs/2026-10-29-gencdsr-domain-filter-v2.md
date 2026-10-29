# Correct domain scope before common-user intersection

The user-authorized GenCDSR-reference SIR experiment produced only13 users from
3908 initial common users. The server ledger establishes normal raw input:
296337 Sports and278677 Clothing events; initial common population79111 events.
The v1 choice of re-pruning item3/user5 inside paired users shrank it through15
deleting passes to213 events. No model or recommendation metric was run on v1.

This is a correction to our reconstruction, not evidence of the author's exact
processor. Use namespace gencdsr-inspired-sc-v2, separate server checkout/study.
Keep v1 checkout/commit dfafeff9 and its diagnosis receipts, and all older runs.

Chosen correction: independently filter each FULL domain to event-occurrence
item>=3/user>=5 fixed points, then intersect retained original users. Remove
non-common users once; NEVER re-prune item degrees in the common subset.
User>=5 in each domain remains true in final shared rows; item>=3 only describes
the full-domain filtering population, and may fail inside the common subset.
For already5-core input this filtering can correctly be a no-op. Do not force
a dataset change merely to make a new protocol differ from the old one.

Alternatives: retaining v1 would train on a tiny unrepresentative task; stopping
its iteration at an arbitrary round would choose a density without upstream
justification. Neither is used. The correction is based on filtering scope and
population, never recommendation outcomes. Counts3908/13057/13044 are expected
only if raw full-domain thresholds truly pass; assert none of these constants
in production. Print measured counts and compare with old population explicitly.

Keep SIR causal per-domain holdouts/source-history visibility, original model,
tokenizer recipe, NDCG10 selection, seeds42/43/44 and GPU admission unchanged.
Do not claim full alignment: author's data table includes non-overlap users and
has different Sports/Clothing sizes. No count matching or threshold HPO.

Audit: bind raw hashes/parser/converter; log full-domain every-pass statistics,
full filtered-domain population, initial overlap and final overlap separately.
Validate final paired user minima and exact rows, without mistakenly applying
global item minima to the overlap. Optional raw-root audit independently replays
full-domain filtering and matches raw hashes and complete filter trace.
Production CLI audit must supply raw root, no GPU involved. Both directions must
have identical provenance/event sets; unchanged assets/freeze identity gates.

Deliver tested source on a new GitHub branch, pinned deployment in
/Data/lihb/SIR-CDR-gencdsr-sc-v2. Original data read from /Data/lihb/SIR-CDR.
No remote execution, no controller resume, no old output overwrite.
