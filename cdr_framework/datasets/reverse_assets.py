"""Independent reverse-direction datasets; forward converters stay byte-for-byte intact."""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import zipfile

from cdr_framework.datasets.amazon2014 import category_files, iter_amazon_records
from cdr_framework.datasets.amazon_preprocessing import (
    AmazonEvent, PreparedDomainPair, prepare_domain_pair, build_temporal_splits, write_preprocessed_artifacts)
from experiments.prepare_gencdr_douban import ARCHIVE_SHA256, REQUIRED, sha256


PAIRS = {
    'clothing_to_sports': ('Clothing_Shoes_and_Jewelry', 'Sports_and_Outdoors', 'sports_to_clothing'),
    'electronics_to_phones': ('Electronics', 'Cell_Phones_and_Accessories', 'phones_to_electronics'),
    'douban_movie_book': ('Movie', 'Book', 'douban_book_movie'),
}


def write_json(path, payload):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    temporary.replace(path)


def _code_identity():
    base = Path(__file__).parent
    return {name: sha256(base / name) for name in ('reverse_assets.py', 'amazon_preprocessing.py', 'amazon2014.py')}


def _recipe(pair, files):
    source, target, forward = PAIRS[pair]
    return {'schema': 'reverse-assets-v1', 'pair': pair, 'source_domain': source, 'target_domain': target,
        'forward_pair': forward, 'minimum_events_per_domain': 5, 'max_text_chars': 12000 if target != 'Book' else 2000,
        'split': 'unchanged Amazon build_temporal_splits; retain repeats; last test, penultimate validation',
        'source_visibility': 'strictly before target label timestamp', 'encoder_history_length': 50,
        'stored_histories': 'full for seen filtering', 'paper_alignment_verified': False,
        'code': _code_identity(), 'raw_files': {p.name: sha256(p) for p in files}}


def verify_reverse(output, expected=None):
    output = Path(output)
    receipt = json.loads((output / 'reverse_manifest.json').read_text(encoding='utf-8'))
    if receipt.get('complete') is not True or (expected is not None and receipt['identity'] != expected):
        raise RuntimeError('Reverse processed identity differs or preparation incomplete')
    required = {'mappings.json', 'item_texts.jsonl', 'train.jsonl', 'validation.jsonl', 'test.jsonl',
                'manifest.json', 'statistics.json', 'protocol.json'}
    if not required.issubset(receipt['files']):
        raise RuntimeError('Reverse processed manifest lacks required files')
    for name, digest in receipt['files'].items():
        if Path(name).name != name or sha256(output / name) != digest:
            raise RuntimeError(f'Reverse processed hash mismatch: {name}')
    return True


def _reuse(output, identity):
    if not output.exists(): return False
    verify_reverse(output, identity)
    print(f'Reusing verified reverse dataset: {output}', flush=True)
    return True


def _finish(prepared, output, identity, raw_files, provenance=None):
    samples = build_temporal_splits(prepared)
    if any(not getattr(samples, name) for name in ('train', 'validation', 'test')):
        raise RuntimeError('Reverse dataset has an empty split; no output accepted')
    write_preprocessed_artifacts(prepared, samples, output, min_interactions=5,
        max_text_chars=identity['max_text_chars'], raw_files=raw_files)
    # Counts include repeated positives rather than silently changing the task.
    counts = {f'{name}_positives_in_seen': sum(r.positive_target_item in r.target_items for r in getattr(samples, name))
              for name in ('train', 'validation', 'test')}
    write_json(output / 'protocol.json', {'identity': identity, 'paper_alignment_verified': False,
        'seen_policy': 'same full-history masking as forward tasks; repeated positives retained',
        'positive_counts': counts})
    if provenance is not None:
        with (output / 'text_provenance.jsonl').open('w', encoding='utf-8') as stream:
            for row in provenance: stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'reverse_manifest.json', {'complete': True, 'identity': identity, 'files': files})
    verify_reverse(output, identity)
    print(json.dumps({'processed': str(output), 'train': len(samples.train), 'validation': len(samples.validation),
                      'test': len(samples.test), **counts}), flush=True)
    return output


def prepare_amazon_reverse(pair, root):
    if pair not in ('clothing_to_sports', 'electronics_to_phones'): raise ValueError('Unknown reverse Amazon task')
    root = Path(root); source, target, _ = PAIRS[pair]
    raw = root / 'data/raw/amazon2014'
    sf, tf = category_files(source, raw), category_files(target, raw)
    files = [sf.reviews, sf.metadata, tf.reviews, tf.metadata]
    missing = [str(p) for p in files if not p.is_file()]
    if missing: raise FileNotFoundError('Require existing original Amazon files: '+str(missing))
    identity = _recipe(pair, files); output = root / 'data/processed' / pair
    if _reuse(output, identity): return output
    prepared = prepare_domain_pair(iter_amazon_records(sf.reviews), iter_amazon_records(tf.reviews),
        iter_amazon_records(sf.metadata), iter_amazon_records(tf.metadata), source_domain=source,
        target_domain=target, min_interactions=5, max_text_chars=12000)
    return _finish(prepared, output, identity, {p.name: p for p in files})


def prepare_douban_reverse(archive_path, output, *, expected_sha256=ARCHIVE_SHA256):
    archive_path, output = Path(archive_path), Path(output)
    if sha256(archive_path) != expected_sha256: raise RuntimeError('Douban archive hash mismatch')
    identity = _recipe('douban_movie_book', [archive_path])
    identity['book_text_policy'] = 'clean labels from reverse-training histories and training positives only; never validation/test reviews or forward Book tags'
    if _reuse(output, identity): return output
    tables = {}; events = {}; labels = {}; rejected = Counter()
    csv.field_size_limit(10*1024*1024)
    with zipfile.ZipFile(archive_path) as archive:
        names = {}
        for name in archive.namelist():
            base = Path(name).name
            if base in REQUIRED:
                if base in names: raise RuntimeError('Duplicate Douban ZIP member')
                names[base] = name
        if set(names) != set(REQUIRED): raise RuntimeError('Missing Douban files')
        def records(base):
            with archive.open(names[base]) as stream:
                yield from csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig'), delimiter='\t')
        def table(base, key):
            values = {}; originals = set()
            for row in records(base):
                uid, original = row.get('UID', '').strip(), row.get(key, '').strip()
                if not uid or not original or uid in values or original in originals:
                    raise RuntimeError('Non-bijective Douban UID mapping')
                values[uid] = row; originals.add(original)
            return values
        users = table('users_cleaned.txt', 'user_id')
        for domain, stem in (('Movie', 'movie'), ('Book', 'book')):
            tables[domain] = table(f'{stem}s_cleaned.txt', f'{stem}_id')
            grouped = defaultdict(list)
            for n, row in enumerate(records(f'{stem}reviews_cleaned.txt'), 1):
                uid, item = row.get('user_id', '').strip(), row.get(f'{stem}_id', '').strip()
                rejected[f'{domain}_raw_events'] += 1
                if uid not in users or item not in tables[domain]:
                    rejected[f'{domain}_unmapped_UID'] += 1; continue
                try: stamp = (datetime.strptime(row.get('time', ''), '%Y-%m-%d')-datetime(1970,1,1)).days*86400
                except (ValueError, TypeError):
                    rejected[f'{domain}_invalid_date'] += 1; continue
                original_user, original_item = users[uid]['user_id'], tables[domain][item][f'{stem}_id']
                grouped[original_user].append((AmazonEvent(original_user, original_item, domain, stamp),
                                              n, row.get('labels', '') or '', item))
                if n % 200000 == 0: print(f'Reverse Douban read {domain}: {n}', flush=True)
            ordered = {u: sorted(rows, key=lambda e: (e[0].timestamp, e[0].asin, e[1])) for u, rows in grouped.items()}
            events[domain] = {u: tuple(e[0] for e in rows) for u, rows in ordered.items()}
            if domain == 'Book': labels = ordered
    retained = sorted(u for u in events['Movie'].keys() & events['Book'].keys()
                      if min(len(events['Movie'][u]), len(events['Book'][u])) >= 5)
    if not retained: raise RuntimeError('No reverse Douban common users')
    user_to_id = {u: i for i, u in enumerate(retained)}
    item_to_id = {}; id_to_item = [None]; item_texts = {}; text_rows = []
    tag_counts = defaultdict(Counter); tag_rows = defaultdict(set)
    # The final eligible training target determines the complete union of Book
    # interactions used in histories OR as supervised training positives.
    # Including training positives is deliberate: they are observed training
    # supervision. The last two (held-out) reviews never contribute labels.
    for user in retained:
        target = labels[user]; first_source = events['Movie'][user][0].timestamp
        eligible = [i for i in range(1, len(target)-2) if first_source < target[i][0].timestamp]
        if eligible:
            for event, row, text, uid in target[:eligible[-1]+1]:
                tags = {v.strip() for v in text.split('|') if v.strip() and '\ufffd' not in v}
                tag_counts[event.asin].update(tags)
                if tags: tag_rows[event.asin].add(f'{user}:{row}')
    static_fields = ('name', 'director', 'writer', 'summary', 'country', 'pubdate', 'language')
    for domain in ('Movie', 'Book'):
        originals = {row[domain.lower()+'_id']: (uid, row) for uid, row in tables[domain].items()}
        items = sorted({e.asin for u in retained for e in events[domain][u]})
        for original in items:
            uid, metadata = originals[original]; item = len(id_to_item)
            item_to_id[f'{domain}:{original}'] = item
            id_to_item.append({'domain': domain, 'asin': original, 'original_id': original, 'raw_UID': uid})
            parts = [f'Domain: {domain}. Catalog identifier: {original}.']; used = []
            if domain == 'Book':
                chosen = sorted(tag_counts[original], key=lambda v: (-tag_counts[original][v], v))[:30]
                if chosen: parts.append('Training interaction tags: '+'; '.join(chosen)); used = ['reverse_training_labels']
            else:
                for key in static_fields:
                    value = ' '.join(metadata.get(key, '').split())
                    if value and '\ufffd' not in value: parts.append(f'{key}: {value}'); used.append(key)
            item_texts[item] = ' '.join(parts)[:2000]
            text_rows.append({'item_id': item, 'fields': used, 'ID_fallback': not used,
                              'reverse_training_target_rows': sorted(tag_rows[original]) if domain == 'Book' else []})
    prepared = PreparedDomainPair('Movie', 'Book', user_to_id, item_to_id, id_to_item, item_texts,
        {u: events['Movie'][u] for u in retained}, {u: events['Book'][u] for u in retained},
        dict(rejected) | {'retained_users': len(retained),
                         'retained_source_items': sum(r['domain']=='Movie' for r in id_to_item[1:]),
                         'retained_target_items': sum(r['domain']=='Book' for r in id_to_item[1:])})
    return _finish(prepared, output, identity, {'douban_archive': archive_path}, text_rows)
