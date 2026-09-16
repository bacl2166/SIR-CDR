# Validation Diagnostics

Run after the first recommendation experiment, using its unchanged YAML,
tokenizer artifacts, and best_model.pt. No retraining or API calls are needed.

```bash
cd /root/autodl-tmp/SIR-CDR
git pull --ff-only origin main
mkdir -p logs
set -o pipefail
python -u experiments/diagnose_recommender.py --config configs/amazon_sports_clothing.yaml --device cuda 2>&1 | tee logs/diagnose_validation.log
```

The report is saved separately as
`artifacts/recommendation/sports_to_clothing/validation_diagnostics.json`.
Existing model checkpoints and test metrics are not overwritten.

- `retrieval_only`: full-target cosine retrieval with the trained temperature.
- `candidate_recall`: fraction of all validation positives retrieved in the
  configured candidate pool (normally 200). Missing positives remain misses.
- `generation_reranked`: the existing retrieval plus generation reranking.
- `training_popularity`: target event frequency from each user's latest
  training prefix and label, avoiding overlapping prefix duplication. Ties
  are resolved by ascending item ID. Validation/test interactions are excluded.
- `positives_in_masked_history`: labels already in the visible target history.
  They are still included in the metric denominator and receive no special
  exception to the history mask.

All methods use the current evaluator's truncated-history exclusion policy.
This is not an all-time seen-item mask. Candidates excluded by that policy
cannot be recommended. Items outside the retrieval pool cannot be rescued
by generation reranking. Retrieval scans the entire target catalog without
random negatives; generation scores only the retrieved pool.

Compare retrieval and reranked NDCG at the same cutoff. If reranking reduces
NDCG, investigate score fusion on validation. If candidate recall is low,
investigate retrieval before the decoder. Compare both with the popularity
baseline. These diagnostics locate bottlenecks; they do not demonstrate that
implicit reasoning or CPF is effective without controlled ablations.
