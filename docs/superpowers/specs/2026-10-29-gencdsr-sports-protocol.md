# GenCDSR-inspired Sports/Clothing common protocol

User selected GenCDSR as the data-protocol reference while keeping the SIR-CDR
backbone and modules. This is a reconstructed common protocol, not import of the
author's Sports/Clothing snapshot (not present in the pinned public repository).

## Frozen first version

- Namespace `gencdsr-inspired-sc-v1`, only sports_to_clothing and clothing_to_sports.
- Input is existing Amazon2014 5-core review files and static metadata; raw hashes
  and parser source hash recorded. No arbitrary subsampling/count matching.
- First retain candidate users shared by true original user IDs. Within this
  paired population jointly prune domain items with fewer than3 events and users
  with fewer than5 events in EITHER domain, repeating to a fixed point. Count
  event occurrences and retain repeated interactions. The original input is
  already5-core, so filtering only before intersection could be a no-op. Scope,
  iteration/order are OUR declared reconstruction choices; the paper does not
  fully specify them. Record raw-domain counts, initial common population, every
  pruning pass, and final5/3 invariants. Items are the full retained-population
  catalog, including heldout-only items; missing text uses static/ID fallback.
- Reuse SIR's causal paired-row contract: per-domain last event test, penultimate
  validation; training all nonempty target prefixes; source training prefix
  excludes its own last2 events and source timestamps must precede the label.
  Audit excluded empty-source rows; never move holdouts or fabricate history.
- Stable time/item/event-key order; domain-qualified item identity. Both forward
  and reverse use the same filtered user/event set. All paired nonempty histories
  remain required; this first version does not claim full upstream population.
- Keep existing SIR Qwen3-0.6B, LoRA, graph/prototypes, CD/SP injectors, reasoning,
  CPF, tokenizer architecture and losses. New catalog means new sealed assets
  and new recommendation checkpoints; pinned model download cache can be reused.
- Existing main363ef0b / server a887bd3 / HPO8644017 stay untouched. New branch
  runs in separate server checkout; all artifacts under the new namespace.
- Keep item-level full-target-catalog retrieval + reranking, full-seen masking,
  one positive per row; report repeated positives masked. Do not copy upstream
  SID-string duplicate-hit behavior. This is NOT upstream decoder reproduction.
- Fixed initial recipe: micro8/accum2/eval8, history50,30epochs/patience5, default
  weights/LR/temp; selection NDCG@10. No hidden HPO. Seeds42/43/44; seed42 pilot
  first, then remaining seeds; test only after all six validation bindings freeze.
- No automatic GPU launch. Explicit physical GPU UUID + idle admission + >=12GiB
  free + >=24GiB host available, sticky training/evaluation affinity and OS locks.
  No process signalling, SSH, source hot swap or old result overwrite.

## Upstream reference

GenCDSR bace0d18f53bfeb56ab9bdd564494055b6c38db7:
https://github.com/Applied-Machine-Learning-Lab/RecSys2026_GenCDSR
Paper https://arxiv.org/html/2607.28659v1 section4.1.1 declares user5/item3 and
chronological mixed sequences. Public data directory only contains Book_Movie.
Public data.py uses train seq[:-2] expanded at every nonempty prefix, validation
label[-2], test label[-1]; stage1 mixed sequences and stage2 domain sequences are
distinct. We do not reproduce its joint-then-domain training stages.

## Contract and provenance

Converter output retains original shared-manifest required files to reuse the
strict verifier/catalog interface. New schema + reference + filter audit + our
converter hash in provenance; paper_alignment_verified=false always. Report
all populations and rows at each filtering stage. Standalone audit must recount
rows, verify mapped identities, label positions, allowed training boundaries,
strict source chronology, exclusion ledger and forward/reverse consistency.
Existing build_shared gains optional schema/upstream and narrowly scoped
protocol_overrides keyword arguments; overrides only describe selection/test/
comparison notes and are identity-bound, never override actual data invariants.
Default protocol semantics stay the same. Training/model files remain untouched.

## Deliverable

CPU-testable converter/auditor, resource-safe staged CLI from raw data through
assets/runtime/smoke/train/validation/freeze/test/status, Chinese runbook, isolated
GitHub branch and optional offline bundle. No quality claim without server runs.
