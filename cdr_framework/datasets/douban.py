"""Auditable Douban Book -> Movie protocol using the existing Amazon split rules.

This is a shared-protocol experiment, not the unpublished GenCDR final split.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import zipfile

from experiments.prepare_gencdr_douban import ARCHIVE_SHA256, SOURCE, PAPER, REQUIRED, sha256


RULES = {'schema': 'douban-common-book-movie-v1', 'minimum_events_per_domain': 5,
         'population': 'common users with mapped UIDs and valid dates',
         'item_identity': 'domain plus original catalog ID, joined through UID',
         'repeats': 'retain all valid interactions, matching the two Amazon tasks',
         'sorting': 'UTC midnight timestamp, original item ID, then original row for exact ties',
         'holdout': 'last target=test, penultimate=validation; never replace ineligible holdouts',
         'source_visibility': 'strictly earlier calendar day than label',
         'history': 'full stored histories; existing encoder uses last 50',
         'catalog': 'all retained target items, including held-out-only items (transductive)',
         'seen_filter': 'full target history; repeated positives stay masked and are counted, matching Amazon',
         'book_text': 'catalog ID plus clean labels visible in at least one training source history',
         'movie_text': 'static name/director/writer/summary/country/pubdate/language only',
         'unknown_date': 'exclude and audit; no imputation', 'max_text_chars': 2000,
         'paper_alignment_verified': False, 'global_time_split': False}


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')


def _identity(archive_digest):
    return {'rules': RULES, 'archive_sha256': archive_digest, 'converter_sha256': sha256(__file__),
            'raw_reader_sha256': sha256(Path(__file__).parents[2] / 'experiments/prepare_gencdr_douban.py')}


def verify_processed(output, expected_identity=None):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('complete') is not True or (expected_identity is not None and manifest['identity'] != expected_identity):
        raise RuntimeError('Douban processed identity differs or conversion is incomplete')
    required = {'train.jsonl', 'validation.jsonl', 'test.jsonl', 'mappings.json', 'item_texts.jsonl',
                'protocol.json', 'statistics.json', 'excluded_events.jsonl', 'excluded_samples.jsonl', 'text_provenance.jsonl'}
    if set(manifest['files']) != required:
        raise RuntimeError('Douban processed file manifest differs')
    for name, digest in manifest['files'].items():
        if sha256(output / name) != digest:
            raise RuntimeError(f'Douban processed file hash mismatch: {name}')
    return True


def convert_archive(archive_path, output, *, expected_sha256=ARCHIVE_SHA256):
    archive_path, output = Path(archive_path), Path(output)
    digest = sha256(archive_path)
    if digest != expected_sha256:
        raise RuntimeError('Douban raw archive hash mismatch')
    identity = _identity(digest)
    if output.exists():
        verify_processed(output, identity)
        print(f'Reusing verified Douban processed assets: {output}', flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    # Failed staging directories are retained for inspection, never swapped into
    # the canonical output. Existing assets are never deleted or force-replaced.
    counts = Counter()
    users = {}; metadata = {}; events = {}
    csv.field_size_limit(10 * 1024 * 1024)
    with zipfile.ZipFile(archive_path) as archive, (stage / 'excluded_events.jsonl').open('w', encoding='utf-8') as excluded:
        names = {}
        for name in archive.namelist():
            base = Path(name).name
            if base in REQUIRED:
                if base in names: raise RuntimeError(f'Duplicate archive member: {base}')
                names[base] = name
        if set(names) != set(REQUIRED): raise RuntimeError('Missing Douban archive members')
        def records(base):
            with archive.open(names[base]) as stream:
                yield from csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig'), delimiter='\t')
        for row in records('users_cleaned.txt'):
            uid, original = row.get('UID', '').strip(), row.get('user_id', '').strip()
            if not uid or not original or uid in users or original in users.values():
                raise RuntimeError('User UID mapping is missing or not one-to-one')
            users[uid] = original
        for domain, stem in (('Book', 'book'), ('Movie', 'movie')):
            table = {}; originals = set()
            for row in records(f'{stem}s_cleaned.txt'):
                uid, original = row.get('UID', '').strip(), row.get(f'{stem}_id', '').strip()
                if not uid or not original or uid in table or original in originals:
                    raise RuntimeError(f'{domain} UID mapping is missing or not one-to-one')
                table[uid] = row; originals.add(original)
            metadata[domain] = table
            grouped = defaultdict(list)
            for index, row in enumerate(records(f'{stem}reviews_cleaned.txt'), 1):
                counts[f'{domain}_raw_events'] += 1
                uid, item = row.get('user_id', '').strip(), row.get(f'{stem}_id', '').strip()
                reason = None
                if uid not in users or item not in table:
                    reason = 'unmapped_user_or_item_UID'
                try:
                    timestamp = (datetime.strptime(row.get('time', ''), '%Y-%m-%d') - datetime(1970, 1, 1)).days * 86400
                except (ValueError, TypeError):
                    reason = reason or 'invalid_date'; timestamp = 0
                if reason:
                    counts[f'{domain}_{reason}'] += 1
                    _line(excluded, {'domain': domain, 'row': index, 'reason': reason, 'user_UID': uid, 'item_UID': item})
                    continue
                grouped[uid].append((timestamp, index, item, row.get('labels', '')))
                if index % 100000 == 0: print(f'Read {domain}: {index} rows', flush=True)
            ordered = {}
            for uid, rows in grouped.items():
                seen = set()
                for event in rows:
                    if event[2] in seen:
                        counts[f'{domain}_repeat_events_retained'] += 1
                    seen.add(event[2])
                ordered[uid] = sorted(rows, key=lambda e: (e[0], table[e[2]][f'{stem}_id'], e[1]))
            events[domain] = ordered
        eligible = sorted([uid for uid in events['Book'].keys() & events['Movie'].keys()
                          if all(len(events[d][uid]) >= 5 for d in ('Book', 'Movie'))],
                          key=lambda uid: users[uid])
        common = set(eligible)
        for domain in events:
            for uid, rows in events[domain].items():
                if uid not in common:
                    counts[f'{domain}_population_excluded_events'] += len(rows)
                    for e in rows: _line(excluded, {'domain': domain, 'user_UID': uid, 'item_UID': e[2],
                                                   'row': e[1], 'reason': 'not_in_frozen_common_user_population'})
    if not eligible: raise RuntimeError('No eligible Douban common users')
    user_to_id = {users[uid]: n for n, uid in enumerate(eligible)}
    source_uids = {e[2] for uid in eligible for e in events['Book'][uid]}
    target_uids = {e[2] for uid in eligible for e in events['Movie'][uid]}
    item_to_id = {}; id_to_item = [None]; id_from_uid = {}
    for domain, uids, field in (('Book', source_uids, 'book_id'), ('Movie', target_uids, 'movie_id')):
        for uid in sorted(uids, key=lambda value: metadata[domain][value][field]):
            original = metadata[domain][uid][field]
            number = len(id_to_item)
            item_to_id[f'{domain}:{original}'] = number; id_from_uid[(domain, uid)] = number
            id_to_item.append({'domain': domain, 'original_id': original, 'raw_UID': uid, 'asin': original})
    _write(stage / 'mappings.json', {'padding_item_id': 0, 'source_domain': 'Book', 'target_domain': 'Movie',
        'user_to_id': user_to_id, 'id_to_user': [users[uid] for uid in eligible],
        'item_to_id': item_to_id, 'id_to_item': id_to_item})
    # A source tag is eligible only if at least one emitted training row can see
    # that event. Validation/test-only source interactions cannot supply text.
    tags = defaultdict(Counter); tag_rows = defaultdict(set)
    with (stage / 'excluded_samples.jsonl').open('w', encoding='utf-8') as excluded:
        for split in ('train', 'validation', 'test'):
            with (stage / f'{split}.jsonl').open('w', encoding='utf-8') as stream:
                for uid in eligible:
                    source, target = events['Book'][uid], events['Movie'][uid]
                    indices = range(len(target)-2) if split == 'train' else [len(target)-(2 if split == 'validation' else 1)]
                    last_visible = []
                    for position in indices:
                        label = target[position]
                        visible = [e for e in source if e[0] < label[0]]
                        if not position or not visible:
                            counts[f'{split}_excluded_samples'] += 1
                            _line(excluded, {'split': split, 'user_UID': uid, 'target_row': label[1],
                                             'reason': 'empty_target_history' if not position else 'empty_visible_source'})
                            continue
                        _line(stream, {'user_id': user_to_id[users[uid]],
                            'source_items': [id_from_uid[('Book', e[2])] for e in visible],
                            'target_items': [id_from_uid[('Movie', e[2])] for e in target[:position]],
                            'positive_target_item': id_from_uid[('Movie', label[2])],
                            'timestamp': label[0], 'maximum_source_timestamp': max(e[0] for e in visible),
                            'target_row': label[1]})
                        counts[f'{split}_rows'] += 1
                        counts[f'{split}_positives_in_seen'] += any(e[2] == label[2] for e in target[:position])
                        last_visible = visible
                    if split == 'train':
                        for e in last_visible:
                            item = id_from_uid[('Book', e[2])]
                            labels = {s.strip() for s in e[3].split('|') if s.strip() and '\ufffd' not in s}
                            tags[item].update(labels)
                            if labels: tag_rows[item].add(f'{uid}:{e[1]}')
            print(f'Douban {split}: {counts[f"{split}_rows"]} rows', flush=True)
    if any(not counts[f'{s}_rows'] for s in ('train', 'validation', 'test')):
        raise RuntimeError('An entire Douban split is empty; inspect excluded sample ledger')
    static_fields = ('name', 'director', 'writer', 'summary', 'country', 'pubdate', 'language')
    with (stage / 'item_texts.jsonl').open('w', encoding='utf-8') as stream, (stage / 'text_provenance.jsonl').open('w', encoding='utf-8') as provenance:
        for item, record in enumerate(id_to_item[1:], 1):
            domain = record['domain']; row = metadata[domain][record['raw_UID']]
            parts = [f'Domain: {domain}. Catalog identifier: {record["original_id"]}.']
            fields = []
            if domain == 'Book':
                chosen = sorted(tags[item], key=lambda tag: (-tags[item][tag], tag))[:30]
                if chosen:
                    parts.append('Training interaction tags: ' + '; '.join(chosen)); fields = ['training_labels']
                    counts['books_with_training_tags'] += 1
            else:
                for key in static_fields:
                    text = ' '.join(row.get(key, '').split())
                    if text and '\ufffd' not in text:
                        parts.append(f'{key}: {text}'); fields.append(key)
                counts['movies_with_static_text'] += bool(fields)
            if not fields: counts[f'{domain}_ID_only_texts'] += 1
            _line(stream, {'item_id': item, 'domain': domain, 'text': ' '.join(parts)[:2000]})
            _line(provenance, {'item_id': item, 'fields': fields, 'raw_member': f'{domain.lower()}s_cleaned.txt',
                              'training_source_rows': sorted(tag_rows[item]) if domain == 'Book' else [],
                              'ID_fallback': not fields})
    counts.update(users=len(eligible), source_items=len(source_uids), target_items=len(target_uids))
    _write(stage / 'statistics.json', dict(counts))
    _write(stage / 'protocol.json', {'identity': identity, 'rules': RULES, 'raw_source': SOURCE, 'baseline_paper': PAPER,
        'paper_alignment_verified': False, 'comparison_scope': 'common-protocol reproduction only',
        'ID_namespace': 'raw reviews use metadata UID, never direct raw catalog-ID joins',
        'source_user_UIDs': eligible, 'source_static_text_missing': True})
    files = {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}
    _write(stage / 'manifest.json', {'complete': True, 'identity': identity, 'files': files})
    verify_processed(stage, identity)
    stage.rename(output)
    print(json.dumps({'processed': str(output), 'statistics': dict(counts), 'paper_alignment_verified': False}), flush=True)
    return output
