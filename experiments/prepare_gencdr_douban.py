"""Fetch and audit original Douban raw data; never substitute GenCDSR splits.

This is a raw-asset audit, not a claim to reproduce GenCDR's paper population.
It deliberately does not launch embeddings, tokenization or recommendation.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'https://github.com/hupeiyu21/GenCDR'
REVISION = '204b0ce0f624d04f84d0c72c52f4d62512ce4bdf'
PAPER = 'https://arxiv.org/html/2511.08006v1'
SOURCE = 'https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information'
DOWNLOAD = ('https://www.kaggle.com/api/v1/datasets/download/fengzhujoey/'
            'douban-datasetratingreviewside-information?datasetVersionNumber=1')
ARCHIVE_SHA256 = 'b1571ebf29c875f5c13bdc1c1eaae092c507c099c2ef62a050f42aab867d10be'
PAPER_COUNTS = {
    'Books': {'users': 1713, 'items': 8601, 'interactions': 104295},
    'Movies': {'users': 2628, 'items': 20964, 'interactions': 1249016},
}
REQUIRED = ('bookreviews_cleaned.txt', 'books_cleaned.txt',
            'moviereviews_cleaned.txt', 'movies_cleaned.txt', 'users_cleaned.txt')


def cleanup_legacy_assets(root):
    """Remove only named legacy GenCDSR assets within this repository.

    Source deletion is handled by Git; this action handles ignored server data.
    Resolve and validate every candidate before removing any of them.
    """
    root = Path(root).resolve()
    locations = ('data/raw', 'data/processed', 'artifacts/embeddings/qwen3_8b',
                 'artifacts/tokenizer/qwen3_8b', 'artifacts/text_cdr_v4/qwen3_8b',
                 'artifacts/llm_cdr_v1', 'artifacts/llm_cdr_v1/qwen3_0_6b',
                 'artifacts/recommendation/qwen3_8b', 'artifacts/benchmark_protocols',
                 'artifacts/retired_drafts')
    targets = [p for location in locations for p in (root / location).glob('gencdsr*')]
    for path in targets:
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f'Unsafe legacy path: {path}')
    removed = []
    for path in targets:
        if not path.exists():
            continue
        print(f'Remove legacy asset: {path}', flush=True)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path.relative_to(root).as_posix())
    return removed


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(path):
    path = Path(path)
    if path.exists():
        if sha256(path) != ARCHIVE_SHA256:
            raise ValueError('Cached archive hash mismatch; existing file was not replaced')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    with urlopen(DOWNLOAD, timeout=90) as source, partial.open('wb') as target:
        size = 0
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > 1024**3:
                raise ValueError('Unexpected archive size')
            target.write(chunk)
            print(f'Douban archive: {size / 1024**2:.1f} MiB', flush=True)
    if sha256(partial) != ARCHIVE_SHA256:
        raise ValueError('Downloaded archive hash mismatch; no assets accepted')
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
    partial.replace(path)
    return path


def audit_reviews(stream, item_column, min_interactions=5):
    """Audit the released processor's per-user filter, without silently fixing dates.

    Unknown dates sort first only for the legacy holdout audit; their presence
    always prevents a paper-alignment claim. CSV order breaks within-day ties.
    No ratings, repeated events or zero-valued original IDs are discarded.
    """
    reader = csv.DictReader(stream, delimiter='\t')
    missing = {'user_id', item_column, 'time'} - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f'Missing review columns: {sorted(missing)}')
    users = defaultdict(list)
    invalid_dates = missing_ids = replacement_characters = 0
    for index, row in enumerate(reader):
        user, item = row.get('user_id'), row.get(item_column)
        if not user or not item:
            missing_ids += 1
            continue
        try:
            date = datetime.strptime(row['time'], '%Y-%m-%d').date().toordinal()
        except (ValueError, TypeError):
            date = 0
            invalid_dates += 1
        replacement_characters += sum((v or '').count('\ufffd') for v in row.values() if isinstance(v, str))
        users[user].append((date, index, item))

    def counts(sequences):
        return {'users': len(sequences),
                'items': len({e[2] for seq in sequences.values() for e in seq}),
                'interactions': sum(map(len, sequences.values()))}

    filtered = {u: sorted(seq) for u, seq in users.items() if len(seq) >= min_interactions}
    repeats = {key: sum(seq[pos][2] in {e[2] for e in seq[:pos]} for seq in filtered.values())
               for key, pos in (('validation', -2), ('test', -1))}
    return {
        'raw': counts(users), 'filtered': counts(filtered),
        'invalid_dates': invalid_dates, 'missing_ids': missing_ids,
        'replacement_characters_in_raw_fields': replacement_characters,
        'same_date_adjacent_pairs': sum(a[0] == b[0] for seq in filtered.values()
                                        for a, b in zip(seq, seq[1:])),
        'holdout_positive_in_history': repeats,
        'paper_alignment_verified': False,
    }, set(filtered)


def audit_archive(path, output, min_interactions=5):
    if min_interactions < 3:
        raise ValueError('min_interactions must be at least 3 for holdout auditing')
    path, output = Path(path), Path(output)
    if sha256(path) != ARCHIVE_SHA256:
        raise ValueError('Archive differs from the audited publisher snapshot')
    reports, user_sets, member_hashes = {}, {}, {}
    with zipfile.ZipFile(path) as archive:
        names = {}
        for name in archive.namelist():
            base = Path(name).name
            if base in REQUIRED:
                if base in names:
                    raise ValueError(f'Duplicate ZIP member: {base}')
                names[base] = name
        if set(names) != set(REQUIRED):
            raise ValueError(f'Missing raw files: {set(REQUIRED) - set(names)}')
        for base, name in names.items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            member_hashes[base] = digest.hexdigest()
        for domain, stem in (('Books', 'book'), ('Movies', 'movie')):
            with archive.open(names[f'{stem}reviews_cleaned.txt']) as stream:
                reports[domain], user_sets[domain] = audit_reviews(
                    io.TextIOWrapper(stream, encoding='utf-8-sig'), f'{stem}_id', min_interactions)
            reports[domain]['paper_counts'] = PAPER_COUNTS[domain]
            reports[domain]['counts_match_paper'] = reports[domain]['filtered'] == PAPER_COUNTS[domain]
    report = {
        'baseline': 'GenCDR', 'upstream': UPSTREAM, 'reference_revision': REVISION,
        'paper': PAPER, 'raw_source': SOURCE, 'archive_sha256': sha256(path),
        'raw_member_sha256': member_hashes, 'audit_code_sha256': sha256(__file__),
        'filter': {'minimum_events_per_user': min_interactions, 'iterative_item_filter': False},
        'domains': reports, 'common_filtered_user_ids': len(user_sets['Books'] & user_sets['Movies']),
        'paper_alignment_verified': False, 'recommendation_assets_ready': False,
        'next_required': ['Verify paper filtered user/item IDs and splits with authors',
                          'Resolve candidate catalog, history visibility and repeated-positive policy',
                          'Build common inputs for both models before a controlled comparison'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('download', 'audit', 'all', 'cleanup-legacy'), default='all')
    parser.add_argument('--archive', type=Path, default=ROOT / 'data/raw/douban_fengzhu_v1.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/gencdr_douban/raw_audit.json')
    parser.add_argument('--min-interactions', type=int, default=5)
    args = parser.parse_args()
    if args.action == 'cleanup-legacy':
        print(json.dumps({'removed': cleanup_legacy_assets(ROOT)}))
        return
    if args.action in ('download', 'all'):
        download_archive(args.archive)
    if args.action in ('audit', 'all'):
        result = audit_archive(args.archive, args.output, args.min_interactions)
        print(json.dumps({'baseline': 'GenCDR', 'report': str(args.output),
                          'domains': {k: v['filtered'] for k, v in result['domains'].items()},
                          'paper_alignment_verified': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
