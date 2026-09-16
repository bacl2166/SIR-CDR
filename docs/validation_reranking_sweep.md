# Validation Candidate and Fusion Sweep

Use the original training YAML and best checkpoint. Only validation metrics
select inference settings; no training or embedding API call is performed.

Score = cosine(query, item) / training_temperature
        + alpha * mean_autoregressive_log_probability(Semantic_ID_and_EOS).

The default sweep uses pools of 200, 500, and 1000 retrieved items and alpha
values 0, 0.5, 1, and 2. Alpha zero measures retrieval-only results and pool
recall. A missing positive is a miss, never inserted into the pool. Masking
matches the current evaluator's truncated target history policy. The entire
target catalog is scanned; generation reranks only the selected pool.

```bash
cd /root/autodl-tmp/SIR-CDR
git pull --ff-only origin main
mkdir -p logs
nohup python -u experiments/sweep_reranking.py --config configs/amazon_sports_clothing.yaml --device cuda --batch-size 16 > logs/sweep_reranking.log 2>&1 &
echo $! > logs/sweep_reranking.pid
tail -f logs/sweep_reranking.log
```

Results go to a separate `validation_sweep_<fingerprint>.json` file in the
recommendation output directory. Each completed group is saved atomically.
Rerunning the same command resumes at the next incomplete group. Changes to
weights, pool sizes, batch size, checkpoint, code, or data create a new report.
Do not edit the original YAML for this sweep.

Selection uses validation NDCG@10, with smaller pools and then smaller alpha
preferred on exact ties. Reports also include HR/NDCG/MRR at 5, 10, and 20,
candidate recall for alpha zero, and seconds per group. For this one-positive
protocol HR@K equals Recall@K. Timings include progress printing and metric
collection, but exclude initial model loading and input hash verification.

Reducing --batch-size to 8 can lower evaluation memory use. This changes the
report fingerprint, so previous groups are not automatically reused. Keep
the existing baseline test result; select settings on validation before any
subsequent final test evaluation. Improvements are not guaranteed.
