"""Immutable evidence and SIR completion gate, independent of SIR model code."""
import math
import statistics
from pathlib import Path

from cdr_framework.datasets.gencdr_protocol import sha
from cdr_framework.experiment_control import write_json_atomic, validate_batch_id
from .assets import read

PAIRS = ('food_to_kitchen', 'kitchen_to_food', 'beauty_to_electronics', 'electronics_to_beauty')
SEEDS = (42, 43, 44)
REVERSE = dict(zip(PAIRS, (PAIRS[1], PAIRS[0], PAIRS[3], PAIRS[2])))
METRICS = tuple(f'{name}@{k}' for k in (5, 10, 20) for name in ('HR', 'NDCG'))


def jobs():
    return [dict(pair=p, seed=s) for s in SEEDS for p in PAIRS]


def key(job):
    return f"{job['pair']}_seed{job['seed']}"


def bind(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'Immutable record changed: {path}')
    else:
        write_json_atomic(path, value)
    return value


def verify_files(root, files):
    root = Path(root).resolve()
    for relative, digest in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
            raise RuntimeError(f'Bound input changed or missing: {relative}')


def metric_key(m):
    values = [float(m[n]) for n in ('NDCG@5', 'NDCG@10', 'HR@5', 'HR@10')]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError('Invalid ranking metrics')
    return ((values[0]+values[1])/2, (values[2]+values[3])/2, values[1])


def sir_completion(root, batch):
    root = Path(root).resolve()
    out = root / 'artifacts/amazon_recommendation' / validate_batch_id(batch)
    result_path, freeze_path = out / 'results.json', out / 'validation_freeze.json'
    if not result_path.exists():
        return None  # A dead controller does NOT mean experiments finished.
    if not freeze_path.is_file():
        raise RuntimeError('SIR result incomplete: validation freeze absent')
    result, freeze = read(result_path), read(freeze_path)
    if (freeze.get('schema') != 'amazon-12-job-validation-freeze-v1'
            or not freeze.get('test_released') or result.get('freeze_sha256') != sha(freeze_path)
            or freeze.get('declaration_sha256') != sha(out / 'declaration.json')):
        raise RuntimeError('SIR freeze/result binding inconsistent')
    expected = {(j['pair'], j['seed']) for j in jobs()}
    for rows in (result.get('per_seed', []), freeze.get('jobs', [])):
        if len(rows) != len(expected) or {(j['pair'], j['seed']) for j in rows} != expected:
            raise RuntimeError('SIR batch incomplete or job identities differ')
    seals = {str(result_path.relative_to(root)): sha(result_path),
             str(freeze_path.relative_to(root)): sha(freeze_path),
             str((out / 'declaration.json').relative_to(root)): sha(out / 'declaration.json')}
    for job in freeze['jobs']:
        if not job.get('files') or not job.get('test_identity'):
            raise RuntimeError('SIR validation provenance absent/incomplete')
        verify_files(root, job['files'])
        run = out / 'jobs' / job['pair'] / f"seed{job['seed']}"
        report_path = run / 'training/test.json'
        receipt_path = out / 'test_completions' / f'{key(job)}.json'
        if not report_path.is_file() or not receipt_path.is_file():
            raise RuntimeError('SIR batch incomplete: test receipt missing')
        report, receipt = read(report_path), read(receipt_path)
        row = next(r for r in result['per_seed'] if (r['pair'], r['seed']) == (job['pair'], job['seed']))
        if (report.get('state') != 'complete' or report.get('split') != 'test'
                or report.get('recovery_identity') != job['test_identity']
                or receipt.get('test_identity') != job['test_identity']
                or receipt.get('freeze_sha256') != sha(freeze_path)
                or receipt.get('report_sha256') != sha(report_path)
                or row['metrics'] != report['metrics']):
            raise RuntimeError('SIR test evidence inconsistent')
        for path in (report_path, receipt_path):
            seals[str(path.relative_to(root))] = sha(path)
    return seals


def summarize(rows):
    summary = []
    for pair in PAIRS:
        group = [r for r in rows if r['pair'] == pair]
        if len(group) != 3 or {r['seed'] for r in group} != set(SEEDS):
            raise RuntimeError('All three seeds required for aggregate')
        summary.append(dict(pair=pair, metrics={n: dict(mean=statistics.mean(r['metrics'][n] for r in group),
            std=statistics.stdev(r['metrics'][n] for r in group)) for n in METRICS}))
    return summary
