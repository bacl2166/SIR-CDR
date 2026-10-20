"""Static-text raw loaders and a timestamp-preserving external sequence bridge."""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import zipfile

from cdr_framework.datasets.gencdr_protocol import Event, sha


PAIRS = {
    'sports_to_clothing': ('Sports_and_Outdoors', 'Clothing_Shoes_and_Jewelry'),
    'clothing_to_sports': ('Clothing_Shoes_and_Jewelry', 'Sports_and_Outdoors'),
    'phones_to_electronics': ('Cell_Phones_and_Accessories', 'Electronics'),
    'electronics_to_phones': ('Electronics', 'Cell_Phones_and_Accessories'),
    'douban_book_movie': ('Book', 'Movie'),
    'douban_movie_book': ('Movie', 'Book'),
}


def raw_amazon(root, source, target):
    from cdr_framework.datasets.amazon2014 import category_files, iter_amazon_records
    from cdr_framework.datasets.amazon_preprocessing import build_product_text
    events = []; items = {}; files = {}; counts = Counter()
    for domain in (source, target):
        paths = category_files(domain, Path(root) / 'data/raw/amazon2014')
        for path in (paths.reviews, paths.metadata): files[path.name] = sha(path)
        wanted = set()
        for n, row in enumerate(iter_amazon_records(paths.reviews), 1):
            user, item, stamp = row.get('reviewerID'), row.get('asin'), row.get('unixReviewTime')
            try:
                if not user or not item or isinstance(stamp, bool): raise ValueError()
                timestamp = int(stamp)
            except (TypeError, ValueError):
                counts[f'{domain}_invalid_records'] += 1; continue
            events.append(Event(str(user), domain, str(item), timestamp, f'{n:012d}'))
            wanted.add(str(item)); counts[f'{domain}_valid_events'] += 1
        for row in iter_amazon_records(paths.metadata):
            item = str(row.get('asin', ''))
            if item in wanted and (domain, item) not in items:
                text = build_product_text(row, 12000)
                if text: items[domain, item] = text
        for item in wanted:
            if (domain, item) not in items:
                items[domain, item] = f'Domain: {domain}. Catalog identifier: {item}.'
                counts[f'{domain}_ID_only_texts'] += 1
    # Deliberately do NOT drop events because static metadata is missing.
    return events, items, {'kind': 'common-protocol-raw', 'raw_files': files,
                          'reader_sha256': sha(__file__), 'raw_audit': dict(counts),
                          'text_policy': 'static catalog attributes; ID fallback; no reviews/ratings/aggregate tags'}


def raw_douban(root):
    from experiments.prepare_gencdr_douban import ARCHIVE_SHA256, REQUIRED
    archive = Path(root) / 'data/raw/douban_fengzhu_v1.zip'
    if sha(archive) != ARCHIVE_SHA256: raise ValueError('Douban publisher archive SHA256 differs')
    events = []; items = {}; counts = Counter()
    csv.field_size_limit(10 * 1024 * 1024)
    with zipfile.ZipFile(archive) as zipped:
        names = {}
        for name in zipped.namelist():
            base = Path(name).name
            if base in REQUIRED:
                if base in names: raise ValueError('Duplicate archive member')
                names[base] = name
        if set(names) != set(REQUIRED): raise ValueError('Missing Douban tables')
        def rows(name):
            with zipped.open(names[name]) as stream:
                yield from csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig'), delimiter='\t')
        def table(name, key):
            values = {}; originals = set()
            for row in rows(name):
                uid, original = row.get('UID', '').strip(), row.get(key, '').strip()
                if not uid or not original or uid in values or original in originals:
                    raise ValueError('Non-bijective UID mapping')
                values[uid] = row; originals.add(original)
            return values
        users = table('users_cleaned.txt', 'user_id')
        for domain, stem in (('Book', 'book'), ('Movie', 'movie')):
            metadata = table(f'{stem}s_cleaned.txt', f'{stem}_id')
            for uid, row in metadata.items():
                original = row[f'{stem}_id']; parts = [f'Domain: {domain}. Catalog identifier: {original}.']
                # This source has no reliable static Book title table. Do not reuse
                # old interaction-derived tags after changing training visibility.
                if domain == 'Movie':
                    for field in ('name', 'director', 'writer', 'summary', 'country', 'pubdate', 'language'):
                        value = ' '.join(row.get(field, '').split())
                        if value and '\ufffd' not in value: parts.append(f'{field}: {value}')
                items[domain, original] = ' '.join(parts)[:2000]
            for n, row in enumerate(rows(f'{stem}reviews_cleaned.txt'), 1):
                uid, item = row.get('user_id', '').strip(), row.get(f'{stem}_id', '').strip()
                try:
                    if uid not in users or item not in metadata: raise ValueError()
                    timestamp = (datetime.strptime(row.get('time', ''), '%Y-%m-%d') - datetime(1970, 1, 1)).days * 86400
                except (ValueError, TypeError):
                    counts[f'{domain}_invalid_records'] += 1; continue
                events.append(Event(users[uid]['user_id'], domain, metadata[item][f'{stem}_id'], timestamp, f'{n:012d}'))
                counts[f'{domain}_valid_events'] += 1
    return events, items, {'kind': 'common-protocol-raw', 'raw_files': {archive.name: ARCHIVE_SHA256},
                          'reader_sha256': sha(__file__), 'raw_audit': dict(counts),
                          'text_policy': 'Movie static whitelist; Book ID-only; no interaction-derived labels'}


def load_bundle(folder, source, target):
    """Load normalized external input, checking native GenCDR sequence boundaries.

    catalog.jsonl: domain, original_id, text. events.jsonl: Event fields.
    sequence_splits.jsonl: user, domain, train/val/test ORIGINAL item-ID lists.
    The final lists must match timestamp order; no guessing history timestamps or
    joining domain-prefixed user numbers. This is deliberately fail-closed.
    """
    folder = Path(folder)
    names = ('protocol.json', 'catalog.jsonl', 'events.jsonl', 'sequence_splits.jsonl')
    metadata = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    if metadata.get('text_policy') != 'static-or-id-only' or not metadata.get('source_description'):
        raise ValueError('External bundle requires static-or-id-only text policy and source description')
    def rows(name):
        with (folder / name).open(encoding='utf-8') as stream:
            for row in stream:
                if row.strip(): yield json.loads(row)
    items = {}
    for row in rows('catalog.jsonl'):
        key = (row['domain'], row['original_id'])
        if key in items: raise ValueError('Duplicate external catalog identity')
        items[key] = row['text']
    events = [Event(**row) for row in rows('events.jsonl')]
    grouped = defaultdict(list)
    for e in sorted(events, key=lambda e: (e.timestamp, e.item, e.key)): grouped[e.user, e.domain].append(e.item)
    seen = set()
    for row in rows('sequence_splits.jsonl'):
        key = (row['user'], row['domain']); seq = grouped.get(key)
        if key in seen or seq is None or row['test'] != seq or row['val'] != seq[:-1] or row['train'] != seq[:-2]:
            raise ValueError('External sequence split/order does not match timed leave-one-out')
        seen.add(key)
    if seen != set(grouped) or {d for _, d in grouped} != {source, target}:
        raise ValueError('External sequence boundaries are incomplete or domains differ')
    # Never silently shrink an author population to meet our cross-domain loader.
    users = {u for u, _ in grouped}
    if any(min(len(grouped.get((u, d), [])) for d in (source, target)) < 5 for u in users):
        raise ValueError('External population contains users unsupported by this paired >=5 protocol; negotiate a shared population first')
    return events, items, {'kind': 'external-sequence-bundle', 'metadata': metadata,
                          'files': {n: sha(folder / n) for n in names}, 'reader_sha256': sha(__file__)}
