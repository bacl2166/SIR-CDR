"""Pure, validation-only policy for the bounded Sports/Clothing study.

No training, scoring, filesystem access or test-split action lives here. Callers
must bind these decisions to their verified validation artifact provenance.
"""
from collections.abc import Mapping
import math
from numbers import Real
import statistics


PAIRS = ("sports_to_clothing", "clothing_to_sports")
SEEDS = (42, 43, 44)
METRICS = ("R@5", "N@5", "R@10", "N@10", "R@20", "N@20")
_PRIORITY = ("N@10", "R@10", "N@5", "R@5")
_CANDIDATES = tuple(f"c{i:02d}" for i in range(8))
_WEIGHTS = (0, .125, .25, .375, .5, .625, .75, .875, 1, 1.25, 1.5, 2)


def candidate_specs():
    """Eight fixed configurations; overrides merge onto original defaults."""
    changes = ({}, {"learning_rate": 3e-5}, {"learning_rate": 3e-4},
               {"retrieval_temperature": .05}, {"retrieval_temperature": .2},
               {"generation_loss_weight": .5}, {"cpf_weight": .03},
               {"learning_rate": 3e-5, "retrieval_temperature": .05,
                "generation_loss_weight": .5, "cpf_weight": .03})
    return [{"id": cid, "overrides": {"selection_metric": "NDCG@10", **change}}
            for cid, change in zip(_CANDIDATES, changes)]


def fusion_settings():
    """The fixed Cartesian grid, using original snapshot-ranker field names."""
    return [{"pool": pool, "weight": weight, "normalization": normalization}
            for pool in (500, 1000) for weight in _WEIGHTS
            for normalization in ("none", "zscore")]


def screen_jobs():
    return [{"pair": pair, "candidate": cid, "seed": 42}
            for cid in _CANDIDATES for pair in PAIRS]


def _promotions(promotions):
    if not isinstance(promotions, Mapping) or set(promotions) != set(PAIRS):
        raise ValueError("Promotions must cover both directions exactly")
    result = {}
    for pair in PAIRS:
        candidates = promotions[pair]
        if (not isinstance(candidates, (list, tuple)) or len(candidates) != 2
                or any(not isinstance(c, str) or c not in _CANDIDATES for c in candidates)
                or len(set(candidates)) != 2):
            raise ValueError("Each direction requires two distinct known candidates")
        result[pair] = sorted(candidates)
    return result


def confirmation_jobs(promotions):
    promotions = _promotions(promotions)
    return [{"pair": pair, "candidate": promotions[pair][position], "seed": seed}
            for position in range(2) for seed in SEEDS[1:] for pair in PAIRS]


def validation_jobs(promotions):
    promotions = _promotions(promotions)
    return [{"pair": pair, "candidate": promotions[pair][position], "seed": seed}
            for position in range(2) for seed in SEEDS for pair in PAIRS]


def _job(row):
    if not isinstance(row, Mapping):
        raise ValueError("Rows must be mappings")
    pair, cid, seed = (row.get(k) for k in ("pair", "candidate", "seed"))
    if (not isinstance(pair, str) or pair not in PAIRS
            or not isinstance(cid, str) or cid not in _CANDIDATES
            or type(seed) is not int or seed not in SEEDS):
        raise ValueError("Unknown direction, candidate or seed")
    return pair, cid, seed


def _metrics(payload, counts, pair):
    if not isinstance(payload, Mapping):
        raise ValueError("Metrics must be a mapping")
    canonical = {}
    for name in METRICS:
        kind, cutoff = name.split('@')
        aliases = (name, f"NDCG@{cutoff}") if kind == 'N' else (
            name, f"HR@{cutoff}", f"Recall@{cutoff}")
        values = [payload[key] for key in aliases if key in payload]
        if not values:
            raise ValueError(f"Missing metric {name}")
        for value in values:
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not 0 <= value <= 1 or not math.isfinite(value)):
                raise ValueError(f"Metric {name} must be finite in [0,1]")
        if any(value != values[0] for value in values):
            raise ValueError(f"Conflicting aliases for {name}")
        canonical[name] = float(values[0])
    if 'examples' in payload:
        count = payload['examples']
        if type(count) is not int or count <= 0:
            raise ValueError("examples must be a positive integer")
        if pair in counts and counts[pair] != count:
            raise ValueError("Inconsistent validation example counts within direction")
        counts[pair] = count
    return canonical


def promote(training_rows):
    """Rank all sixteen native seed42 screens and promote two per direction."""
    observed, counts = {}, {}
    for row in training_rows:
        key = _job(row)
        if key[2] != 42 or key in observed:
            raise ValueError("Screens require distinct seed42 jobs")
        observed[key] = _metrics(row.get('metrics'), counts, key[0])
    expected = {(pair, cid, 42) for pair in PAIRS for cid in _CANDIDATES}
    if set(observed) != expected:
        raise ValueError("Promotion requires all sixteen completed screens")
    return {pair: sorted(_CANDIDATES, key=lambda cid: (
                *(-observed[pair, cid, 42][m] for m in _PRIORITY), cid))[:2]
            for pair in PAIRS}


def _setting(setting):
    if not isinstance(setting, Mapping) or set(setting) != {'pool', 'weight', 'normalization'}:
        raise ValueError("Fusion setting must contain pool, weight, normalization only")
    pool, weight, norm = (setting[k] for k in ('pool', 'weight', 'normalization'))
    if (type(pool) is not int or pool not in (500, 1000)
            or isinstance(weight, bool) or not isinstance(weight, Real)
            or weight not in _WEIGHTS or not isinstance(norm, str)
            or norm not in ('none', 'zscore')):
        raise ValueError("Fusion setting is outside the fixed grid")
    return pool, weight, norm


def select_final(validation_rows):
    """Choose one candidate and common fusion per pair by three-seed means.

    Returns every candidate/setting summary as well as each winning direction.
    Sample standard deviations use n-1; per-seed results preserve all six metrics.
    Every model must supply the full 48-setting grid before any choice is made.
    """
    observed, counts, candidates = {}, {}, {pair: set() for pair in PAIRS}
    expected_settings = {_setting(s) for s in fusion_settings()}
    for row in validation_rows:
        key = _job(row)
        if key in observed:
            raise ValueError("Duplicate finalist job")
        results = row.get('results')
        if not isinstance(results, (list, tuple)):
            raise ValueError("Each finalist needs a list of 48 validation results")
        settings = {}
        for result in results:
            if not isinstance(result, Mapping):
                raise ValueError("Validation result must be a mapping")
            setting_key = _setting(result.get('setting'))
            if setting_key in settings:
                raise ValueError("Duplicate fusion setting")
            settings[setting_key] = _metrics(result.get('metrics'), counts, key[0])
        if set(settings) != expected_settings:
            raise ValueError("Each finalist requires all 48 distinct fusion settings")
        observed[key] = settings
        candidates[key[0]].add(key[1])
    if any(len(candidates[pair]) != 2 for pair in PAIRS):
        raise ValueError("Final selection requires exactly two candidates per direction")
    expected_jobs = {(pair, cid, seed) for pair in PAIRS
                     for cid in candidates[pair] for seed in SEEDS}
    if set(observed) != expected_jobs:
        raise ValueError("Final selection requires all twelve three-seed finalist jobs")
    summaries = []
    for pair in PAIRS:
        for cid in sorted(candidates[pair]):
            for setting in fusion_settings():
                skey = _setting(setting)
                per_seed = [{"seed": seed, "metrics": dict(observed[pair, cid, seed][skey])}
                            for seed in SEEDS]
                aggregate = {}
                for metric in METRICS:
                    values = [row['metrics'][metric] for row in per_seed]
                    aggregate[metric] = {"mean": statistics.mean(values),
                                         "std": statistics.stdev(values)}
                summaries.append({"pair": pair, "candidate": cid, "setting": setting,
                                  "metrics": aggregate, "seeds": list(SEEDS),
                                  "per_seed": per_seed})
    def rank(summary):
        setting = summary['setting']
        return (*(-summary['metrics'][m]['mean'] for m in _PRIORITY),
                summary['candidate'], setting['pool'], setting['weight'],
                0 if setting['normalization'] == 'none' else 1)
    return {"directions": [min((s for s in summaries if s['pair'] == pair), key=rank)
                            for pair in PAIRS], "all_summaries": summaries}
