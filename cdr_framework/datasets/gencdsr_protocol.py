"""Our GenCDSR-inspired Sports/Clothing reconstruction; no author assets claim.

All filtering counts event occurrences, including repeated user/item interactions.
This module deliberately has no torch/CUDA dependency.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from . import gencdr_protocol as shared
from .gencdr_inputs import PAIRS as RAW_PAIRS, raw_amazon

SCHEMA = 'gencdsr-inspired-sc-v2'
PAIRS = ('sports_to_clothing', 'clothing_to_sports')
DOMAINS = {pair: RAW_PAIRS[pair] for pair in PAIRS}
UPSTREAM = {
    'repository': 'https://github.com/Applied-Machine-Learning-Lab/RecSys2026_GenCDSR',
    'revision': 'bace0d18f53bfeb56ab9bdd564494055b6c38db7',
    'paper': 'https://arxiv.org/html/2607.28659v1',
    'split_function': 'data.py: chronological prefix expansion (inspiration only)',
    'author_sports_clothing_assets_available': False,
}
FILTER_SCOPE = 'OUR reconstruction choice: independent full-domain item3/user5 fixed points, then common users once; no common-subset item re-pruning'
RULES = {
    'reconstruction_notes': FILTER_SCOPE,
    'comparison_scope': 'GenCDSR-inspired Sports/Clothing common protocol; unchanged SIR modules; not exact author assets',
    'checkpoint_selection': 'NDCG@10',
    'test_policy': 'test after immutable validation freeze of both pairs x seeds 42/43/44; fixed hybrid/500/0.5/none; no fusion search',
}


def _snapshot(events, domains):
    return {'users': len({e.user for e in events}), 'events': len(events),
            'domains': {d: {'users': len({e.user for e in events if e.domain == d}),
                            'items': len({e.item for e in events if e.domain == d}),
                            'events': sum(e.domain == d for e in events)} for d in sorted(domains)}}


def _ordered_events(events, source, target):
    domains = (source, target)
    ordered = sorted(events, key=lambda e: (e.user, e.domain, e.timestamp, e.item, e.key))
    seen = set()
    for e in ordered:
        if (e.domain not in domains or type(e.timestamp) is not int or
                not all(isinstance(x, str) and x for x in (e.user, e.domain, e.item, e.key))):
            raise ValueError('Invalid interaction or domain')
        key = (e.user, e.domain, e.key)
        if key in seen: raise ValueError('Duplicate event key; repeats require distinct identities')
        seen.add(key)
    return ordered


def _common(events, source, target):
    users = {e.user for e in events if e.domain == source} & {e.user for e in events if e.domain == target}
    return [e for e in events if e.user in users]


def _filter(events, source, target):
    domains = (source, target)
    ordered = _ordered_events(events, source, target)
    current = ordered
    trace = {'policy': FILTER_SCOPE, 'user_min_events_each_domain': 5, 'item_min_events': 3,
             'raw': _snapshot(ordered, domains),
             'initial_common': _snapshot(_common(ordered, source, target), domains), 'iterations': []}
    while True:
        before = _snapshot(current, domains)
        item_counts = Counter((e.domain, e.item) for e in current)
        item_kept = [e for e in current if item_counts[e.domain, e.item] >= 3]
        counts = Counter((e.user, e.domain) for e in item_kept)
        # Each domain is filtered independently. A user's removal in one domain
        # must not remove their support for an item in the other domain.
        retained = [e for e in item_kept if counts[e.user, e.domain] >= 5]
        after = _snapshot(retained, domains)
        trace['iterations'].append({'before': before, 'after_item': _snapshot(item_kept, domains),
                                    'after': after, 'removed_item_events': len(current)-len(item_kept),
                                    'removed_user_events': len(item_kept)-len(retained)})
        current = retained
        if before == after: break
    trace['filtered_domains'] = _snapshot(current, domains)
    common = _common(current, source, target)
    trace['final'] = _snapshot(common, domains)
    if not common: raise ValueError('Full-domain 5/3 filtering has no common users')
    return common, trace


def build_pair(events, items, source, target, output, *, provenance):
    """Filter full domains before intersecting users; retain causal paired rows."""
    if (source, target) not in DOMAINS.values(): raise ValueError('Only Sports/Clothing directions are supported')
    filtered, trace = _filter(events, source, target)
    catalog = {(e.domain, e.item) for e in filtered}
    texts = {key: items.get(key) for key in catalog}
    for (domain, item), text in texts.copy().items():
        if not isinstance(text, str) or not text.strip():
            texts[domain, item] = f'Domain: {domain}. Catalog identifier: {item}.'
    enriched = {**provenance, 'kind': 'gencdsr-inspired-common-protocol', 'filter_audit': trace,
                'upstream_reference': UPSTREAM, 'converter_sha256': shared.sha(__file__),
                'paper_alignment_verified': False,
                'reconstruction_scope': 'OUR full-domain independent 5/3 fixed points then overlap; not exact author assets'}
    return shared.build_shared(filtered, texts, source, target, output, provenance=enriched,
                               schema=SCHEMA, upstream=UPSTREAM, protocol_overrides=RULES)


def convert(root, pair, *, output_root=None):
    """Read raw files from root; output_root is a separate checkout when supplied."""
    if pair not in DOMAINS: raise ValueError(f'Unsupported GenCDSR-inspired pair: {pair}')
    source, target = DOMAINS[pair]
    events, items, provenance = raw_amazon(Path(root), source, target)
    output = Path(output_root if output_root is not None else root)/'data/processed'/SCHEMA/pair
    return build_pair(events, items, source, target, output, provenance=provenance)


def _json(root, name):
    return json.loads((root/name).read_text(encoding='utf-8'))


def _rows(root, name):
    with (root/name).open(encoding='utf-8') as stream:
        for row in stream:
            yield json.loads(row)


def _equal(actual, expected, name):
    if actual != expected: raise ValueError(f'Protocol recount mismatch: {name}')


def _check_rows(root, name, expected):
    from itertools import zip_longest
    sentinel = object()
    for actual, wanted in zip_longest(_rows(root, name), expected, fillvalue=sentinel):
        _equal(actual, wanted, name)


def audit(processed, *, raw_root=None):
    """Verify hashes AND reconstruct every mapping, split row and exclusion.

    raw_root replays full-domain filtering from hashed original files. Without
    it, inspect the bound trace and recount final rows only: global item degrees
    cannot be inferred from the smaller common-user subset.
    """
    root = Path(processed)
    shared.verify_shared(root)
    receipt = _json(root, 'shared_manifest.json'); identity = receipt['identity']
    rules = _json(root, 'protocol.json')
    _equal(rules['identity'], identity, 'identity')
    _equal(identity['schema'], SCHEMA, 'schema'); _equal(rules['schema'], SCHEMA, 'schema')
    _equal(rules.get('paper_alignment_verified'), False, 'alignment')
    _equal(rules.get('upstream_split_reference'), UPSTREAM, 'upstream')
    for key, value in RULES.items(): _equal(rules.get(key), value, key)
    source, target = identity['source_domain'], identity['target_domain']
    if (source, target) not in DOMAINS.values(): raise ValueError('Invalid pair identity')
    _equal(identity['training_examples'], 'all-prefixes', 'training expansion')
    _equal(identity.get('protocol_overrides'), RULES, 'bound documentation overrides')
    _equal(rules.get('source_visibility'), 'source seq[:-2], timestamp < label timestamp', 'source visibility')
    _equal(rules.get('target_split'), 'test=seq; val=seq[:-1]; train=seq[:-2]; final element is the label', 'target split')
    provenance = identity['provenance']
    _equal(provenance.get('converter_sha256'), shared.sha(__file__), 'converter source')
    _equal(provenance.get('upstream_reference'), UPSTREAM, 'reference provenance')
    _equal(provenance.get('paper_alignment_verified'), False, 'provenance alignment')
    events = [shared.Event(**e) for e in _rows(root, 'events.jsonl')]
    _equal(events, _ordered_events(events, source, target), 'canonical filtered events')
    user_counts = Counter((e.user, e.domain) for e in events)
    if not events or any(user_counts[u, d] < 5 for u in {e.user for e in events} for d in (source, target)):
        raise ValueError('Final common users require five events in each domain')
    digest = hashlib.sha256()
    groups = {source: defaultdict(list), target: defaultdict(list)}
    for e in events:
        digest.update(json.dumps(e.__dict__, sort_keys=True).encode())
        groups[e.domain][e.user].append(e)
    _equal(digest.hexdigest(), identity['events_sha256'], 'events digest')
    trace = provenance['filter_audit']
    _equal(trace['final'], _snapshot(events, (source, target)), 'final filter population')
    _equal(trace['policy'], FILTER_SCOPE, 'filter policy')
    _equal(trace['user_min_events_each_domain'], 5, 'user minimum')
    _equal(trace['item_min_events'], 3, 'item minimum')
    previous = trace['raw']
    for step in trace['iterations']:
        _equal(step['before'], previous, 'filter trace continuity')
        if step['before']['events']-step['after_item']['events'] != step['removed_item_events']:
            raise ValueError('Malformed item exclusion counts')
        if step['after_item']['events']-step['after']['events'] != step['removed_user_events']:
            raise ValueError('Malformed user exclusion counts')
        for field in ('users', 'events'):
            if not 0 <= step['after'][field] <= step['after_item'][field] <= step['before'][field]:
                raise ValueError('Nonmonotonic filter trace')
        previous = step['after']
    if not trace['iterations']: raise ValueError('Missing fixed point evidence')
    _equal(previous, trace['filtered_domains'], 'filter trace full-domain final')
    _equal(trace['iterations'][-1]['before'], previous, 'stable terminal pass')
    for field in ('users', 'events'):
        if not 0 < trace['final'][field] <= trace['initial_common'][field] <= trace['raw'][field]:
            raise ValueError('Invalid common-user population trace')
        if trace['final'][field] > trace['filtered_domains'][field]:
            raise ValueError('Common subset exceeds filtered domain population')
    if raw_root is not None:
        raw_events, _, raw_provenance = raw_amazon(Path(raw_root), source, target)
        for key in ('raw_files', 'reader_sha256', 'raw_audit'):
            _equal(raw_provenance[key], provenance[key], 'raw replay '+key)
        replayed, replay_trace = _filter(raw_events, source, target)
        _equal(trace, replay_trace, 'raw filter replay trace')
        _equal(events, replayed, 'raw filter replay events')
    users = sorted(groups[source]); catalog = sorted({(e.domain, e.item) for e in events})
    number = {key: n for n, key in enumerate(catalog, 1)}
    mapping = {'padding_item_id': 0, 'source_domain': source, 'target_domain': target,
               'user_to_id': {u: n for n, u in enumerate(users)}, 'id_to_user': users,
               'item_to_id': {f'{d}:{i}': n for (d, i), n in number.items()},
               'id_to_item': [None]+[{'domain': d, 'asin': i, 'original_id': i} for d, i in catalog]}
    _equal(_json(root, 'mappings.json'), mapping, 'mappings')
    texts = list(_rows(root, 'item_texts.jsonl'))
    if len(texts) != len(catalog): raise ValueError('Incomplete item texts')
    for row, ((domain, item), n) in zip(texts, number.items()):
        if row.get('item_id') != n or row.get('domain') != domain or not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Invalid static item text mapping')
    text_digest = hashlib.sha256(json.dumps([(d, i, row['text']) for (d, i), row in zip(catalog, texts)], ensure_ascii=False).encode()).hexdigest()
    _equal(text_digest, identity['texts_sha256'], 'texts digest')
    counts = Counter(users=len(users), source_items=sum(d == source for d, _ in catalog), target_items=sum(d == target for d, _ in catalog))
    exclusions = []; accepted = {u: {} for u in users}
    def expected_rows(split):
        for u in users:
            targets = groups[target][u]; source_train = groups[source][u][:-2]
            positions = range(1, len(targets)-2) if split == 'train' else [len(targets)-(2 if split == 'validation' else 1)]
            accepted[u][split] = []
            for p in positions:
                label = targets[p]; visible = [e for e in source_train if e.timestamp < label.timestamp]
                if not visible:
                    counts[f'{split}_excluded_empty_source'] += 1
                    exclusions.append({'user': u, 'split': split, 'target_event_key': label.key, 'reason': 'empty_source_training_prefix_before_label'})
                    continue
                accepted[u][split].append(p)
                row = {'user_id': mapping['user_to_id'][u], 'source_items': [number[source, e.item] for e in visible],
                       'target_items': [number[target, e.item] for e in targets[:p]],
                       'positive_target_item': number[target, label.item], 'timestamp': label.timestamp,
                       'target_event_key': label.key, 'target_position': p,
                       'maximum_source_timestamp': max(e.timestamp for e in visible)}
                counts[f'{split}_rows'] += 1
                counts[f'{split}_positives_in_seen'] += row['positive_target_item'] in row['target_items']
                yield row
    for split in ('train', 'validation', 'test'):
        _check_rows(root, f'{split}.jsonl', expected_rows(split))
        if not counts[f'{split}_rows']: raise ValueError('Empty shared split')
    _check_rows(root, 'excluded_samples.jsonl', exclusions)
    bridges = []
    for u in users:
        seq = [number[target, e.item] for e in groups[target][u]]
        bridges.append({'user': u, 'user_id': mapping['user_to_id'][u], 'train': seq[:-2], 'val': seq[:-1], 'test': seq,
                        'accepted_target_positions': accepted[u], 'baseline_must_consume_shared_rows': True})
    _check_rows(root, 'gencdr_sequence_splits.jsonl', bridges)
    counts['source_population_excluded_events'] = counts['target_population_excluded_events'] = 0
    _equal(_json(root, 'statistics.json'), dict(counts), 'statistics')
    return {'complete': True, 'schema': SCHEMA, 'processed': str(root), **counts,
            'events': len(events), 'paper_alignment_verified': False,
            'raw_filter_replayed': raw_root is not None}


def audit_pair_roots(forward, reverse):
    reports = [audit(forward), audit(reverse)]
    roots = [Path(forward), Path(reverse)]
    identities = [_json(root, 'shared_manifest.json')['identity'] for root in roots]
    _equal((identities[0]['source_domain'], identities[0]['target_domain']),
           (identities[1]['target_domain'], identities[1]['source_domain']), 'reverse direction')
    for key in ('events_sha256', 'texts_sha256'):
        _equal(identities[0][key], identities[1][key], 'cross-direction '+key)
    mappings = [_json(root, 'mappings.json') for root in roots]
    for key in ('id_to_user', 'user_to_id', 'id_to_item', 'item_to_id'):
        _equal(mappings[0][key], mappings[1][key], 'cross-direction '+key)
    _equal(identities[0]['provenance']['filter_audit'], identities[1]['provenance']['filter_audit'], 'cross-direction filter audit')
    return {'complete': True, 'schema': SCHEMA, 'users': reports[0]['users'], 'events': reports[0]['events'], 'directions': reports}
