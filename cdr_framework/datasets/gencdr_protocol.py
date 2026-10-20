"""Shared, timed leave-one-out protocol; never claims unavailable author splits.

GenCDR AbstractDataset._leave_one_out exposes train=seq[:-2], val=seq[:-1],
test=seq. This adapter expands only the training prefix into next-item examples.
Source-domain validation/test events are withheld in BOTH directions, including
when their timestamps happen to precede another domain's training target.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile


SCHEMA = 'gencdr-shared-loo-v1'
UPSTREAM = {'repository': 'https://github.com/hupeiyu21/GenCDR',
            'revision': '204b0ce0f624d04f84d0c72c52f4d62512ce4bdf',
            'split_function': 'dataloader/dataset.py:AbstractDataset._leave_one_out'}
REQUIRED = {'mappings.json', 'item_texts.jsonl', 'train.jsonl', 'validation.jsonl',
            'test.jsonl', 'statistics.json', 'protocol.json', 'events.jsonl',
            'excluded_samples.jsonl', 'gencdr_sequence_splits.jsonl'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')


def line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))+'\n')


@dataclass(frozen=True)
class Event:
    user: str
    domain: str
    item: str
    timestamp: int
    key: str  # unique within user/domain; identifies repeats, not an item ID


def verify_shared(output, identity=None):
    output = Path(output)
    receipt = json.loads((output / 'shared_manifest.json').read_text(encoding='utf-8'))
    if receipt.get('complete') is not True or (identity is not None and receipt['identity'] != identity):
        raise RuntimeError('Shared protocol identity changed/incomplete; choose a new output namespace')
    if set(receipt['files']) != REQUIRED:
        raise RuntimeError('Shared manifest has missing or unexpected files')
    for name, digest in receipt['files'].items():
        if Path(name).name != name or sha(output / name) != digest:
            raise RuntimeError(f'Shared protocol hash mismatch: {name}')
    return True


def build_shared(events, items, source, target, output, *, provenance, training_examples='all-prefixes'):
    """Freeze the common population; no random sampling/count matching/holdout replacement.

    `items` maps (domain, original ID) to static/ID-only text. Interaction-derived
    metadata must NOT be passed; raw loaders below enforce this. The normalized
    external bundle is an explicit data contract, not automatically trusted as
    the author's final assets.
    """
    if source == target or training_examples not in ('all-prefixes', 'last-prefix'):
        raise ValueError('Distinct domains and a supported training expansion are required')
    groups = {source: defaultdict(list), target: defaultdict(list)}
    keys = set(); event_digest = hashlib.sha256()
    # Canonicalize before hashing so input iteration order cannot change splits.
    ordered = sorted(events, key=lambda e: (e.user, e.domain, e.timestamp, e.item, e.key))
    for e in ordered:
        if (e.domain not in groups or (e.domain, e.item) not in items or type(e.timestamp) is not int
                or not all(isinstance(x, str) and x for x in (e.user, e.domain, e.item, e.key))):
            raise ValueError('Unmapped or invalid interaction')
        key = (e.user, e.domain, e.key)
        if key in keys: raise ValueError('Duplicate event key; repeats need distinct event identities')
        keys.add(key); groups[e.domain][e.user].append(e)
        event_digest.update(json.dumps(e.__dict__, sort_keys=True).encode())
    population = sorted(u for u in groups[source].keys() & groups[target].keys()
                        if min(len(groups[d][u]) for d in groups) >= 5)
    if not population: raise ValueError('Shared common-user population is empty')
    # Do not let metadata availability choose the interaction population.
    catalog = sorted({(e.domain, e.item) for d in groups for u in population for e in groups[d][u]})
    text_digest = hashlib.sha256(json.dumps([(d, i, items[d, i]) for d, i in catalog],
                                            ensure_ascii=False).encode()).hexdigest()
    identity = {'schema': SCHEMA, 'source_domain': source, 'target_domain': target,
                'events_sha256': event_digest.hexdigest(), 'texts_sha256': text_digest,
                'provenance': provenance, 'training_examples': training_examples,
                'converter_sha256': sha(__file__)}
    output = Path(output)
    if output.exists():
        verify_shared(output, identity); return output
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    number = {item: i for i, item in enumerate(catalog, 1)}
    mapping = {'padding_item_id': 0, 'source_domain': source, 'target_domain': target,
               'user_to_id': {u: i for i, u in enumerate(population)}, 'id_to_user': population,
               'item_to_id': {f'{d}:{item}': n for (d, item), n in number.items()},
               'id_to_item': [None]+[{'domain': d, 'asin': item, 'original_id': item} for d, item in catalog]}
    if len(mapping['item_to_id']) != len(catalog): raise ValueError('Ambiguous domain/item namespace')
    write(stage / 'mappings.json', mapping)
    counts = Counter(users=len(population), source_items=sum(d == source for d, _ in catalog),
                     target_items=sum(d == target for d, _ in catalog))
    with (stage / 'item_texts.jsonl').open('w', encoding='utf-8') as stream:
        for (d, item), n in number.items():
            text = items[d, item]
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Every catalog item needs static text or an explicit ID fallback')
            line(stream, {'item_id': n, 'domain': d, 'text': text})
    with (stage / 'events.jsonl').open('w', encoding='utf-8') as stream:
        for e in ordered: line(stream, e.__dict__)
    with (stage / 'gencdr_sequence_splits.jsonl').open('w', encoding='utf-8') as exported:
        for u in population:
            target_events = groups[target][u]
            seq = [number[target, e.item] for e in target_events]
            first_source = groups[source][u][0].timestamp
            train_positions = (range(1, len(seq)-2) if training_examples == 'all-prefixes' else [len(seq)-3])
            accepted = {'train': [i for i in train_positions if first_source < target_events[i].timestamp],
                        'validation': [len(seq)-2] if first_source < target_events[-2].timestamp else [],
                        'test': [len(seq)-1] if first_source < target_events[-1].timestamp else []}
            line(exported, {'user': u, 'user_id': mapping['user_to_id'][u],
                            'train': seq[:-2], 'val': seq[:-1], 'test': seq,
                            'accepted_target_positions': accepted,
                            'baseline_must_consume_shared_rows': True})
    with (stage / 'excluded_samples.jsonl').open('w', encoding='utf-8') as excluded:
        for split in ('train', 'validation', 'test'):
            with (stage / f'{split}.jsonl').open('w', encoding='utf-8') as stream:
                for u in population:
                    targets = groups[target][u]
                    source_train = groups[source][u][:-2]
                    indices = (range(1, len(targets)-2) if training_examples == 'all-prefixes'
                               else [len(targets)-3]) if split == 'train' else [len(targets)-(2 if split == 'validation' else 1)]
                    # Earlier target prefix and STRICTLY earlier source training events only.
                    for p in indices:
                        label = targets[p]; visible = [e for e in source_train if e.timestamp < label.timestamp]
                        if not visible:
                            counts[f'{split}_excluded_empty_source'] += 1
                            line(excluded, {'user': u, 'split': split, 'target_event_key': label.key,
                                            'reason': 'empty_source_training_prefix_before_label'})
                            continue
                        history = [number[target, e.item] for e in targets[:p]]
                        row = {'user_id': mapping['user_to_id'][u], 'source_items': [number[source, e.item] for e in visible],
                               'target_items': history, 'positive_target_item': number[target, label.item],
                               'timestamp': label.timestamp, 'target_event_key': label.key,
                               'target_position': p, 'maximum_source_timestamp': max(e.timestamp for e in visible)}
                        line(stream, row); counts[f'{split}_rows'] += 1
                        counts[f'{split}_positives_in_seen'] += row['positive_target_item'] in history
    if any(not counts[f'{s}_rows'] for s in ('train', 'validation', 'test')):
        raise ValueError('A shared split is empty; inspect retained staging audit (no holdout substitution)')
    population_set = set(population)
    for name in ('source', 'target'):
        domain = source if name == 'source' else target
        counts[f'{name}_population_excluded_events'] = sum(len(rows) for u, rows in groups[domain].items() if u not in population_set)
    rules = {'schema': SCHEMA, 'identity': identity, 'upstream_split_reference': UPSTREAM, 'paper_alignment_verified': False,
             'comparison_scope': 'rerun both systems on this frozen common protocol; not a paper-table reproduction',
             'population': 'common users with >=5 valid events in EACH domain',
             'target_split': 'test=seq; val=seq[:-1]; train=seq[:-2]; final element is the label',
             'training_examples': training_examples, 'source_visibility': 'source seq[:-2], timestamp < label timestamp',
             'tie_order': 'timestamp, original item ID, event key (lexicographic)', 'repeat_events': 'retained',
             'empty_history': 'exclude and audit, never move holdouts',
             'catalog': 'all retained-population items, including held-out-only items; transductive',
             'seen_policy': 'mask full target history; repeated positives remain masked and counted',
             'metrics': 'one positive per row; R@K=HR@K; N@K=1/log2(rank+1) on hits',
             'candidate_policy': 'full target catalog, no sampled negatives',
             'encoder_history': 50, 'checkpoint_selection': 'HR@10 (equals Recall@10)',
             'test_policy': '17 preset validation-only fusion settings; protect four anchor metrics then maximize N@10; exact verification before immutable test'}
    write(stage / 'protocol.json', rules); write(stage / 'statistics.json', dict(counts))
    write(stage / 'shared_manifest.json', {'complete': True, 'identity': identity,
                                          'files': {n: sha(stage / n) for n in REQUIRED}})
    verify_shared(stage, identity); stage.rename(output)
    print(json.dumps({'processed': str(output), **counts, 'paper_alignment_verified': False}), flush=True)
    return output
