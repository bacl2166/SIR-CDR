import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from cdr_framework.datasets.douban import convert_archive, verify_processed
from experiments.prepare_gencdr_douban import sha256


def raw_fixture(root, target_repeat=False):
    archive = root / 'raw.zip'
    def tsv(rows):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
        return stream.getvalue()
    books = [{'UID': str(10+i), 'book_id': str(42+i)} for i in range(5)]
    movies = [{'UID': str(20+i), 'movie_id': str(42+i), 'name': f'Movie {i}',
               'summary': 'Static story', 'rating': 'FORBIDDEN_RATING', 'tag': 'FORBIDDEN_TAG'} for i in range(5)]
    sources = [{'user_id': '7', 'book_id': str(10+i), 'time': f'2020-01-{day:02}',
                'labels': 'TRAIN_TAG' if day < 10 else 'FUTURE_LABEL', 'ID': str(i)}
               for i, day in enumerate((1, 2, 3, 7, 20))]
    sources += [dict(sources[0], time='2020-01-04', ID='duplicate'),
                dict(sources[0], time='bad', ID='bad')]
    targets = [{'user_id': '7', 'movie_id': str(20+i), 'time': f'2020-01-{day:02}',
                'labels': 'TARGET_LABEL_MUST_NOT_ENTER_TEXT', 'ID': str(i)}
               for i, day in enumerate((5, 6, 8, 9, 10))]
    if target_repeat:
        targets.append(dict(targets[0], time='2020-01-11', ID='repeat_holdout'))
    with zipfile.ZipFile(archive, 'w') as z:
        for name, rows in {'books': books, 'movies': movies,
                           'users': [{'UID': '7', 'user_id': '800'}],
                           'bookreviews': sources, 'moviereviews': targets}.items():
            z.writestr(f'raw/{name}_cleaned.txt', tsv(rows))
    return archive


class DoubanPipelineTests(unittest.TestCase):
    def test_uid_join_splits_and_training_only_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); raw = raw_fixture(root); output = root / 'processed'
            convert_archive(raw, output, expected_sha256=sha256(raw))
            mappings = json.loads((output / 'mappings.json').read_text())
            self.assertNotEqual(mappings['item_to_id']['Book:42'], mappings['item_to_id']['Movie:42'])
            self.assertEqual(mappings['user_to_id'], {'800': 0})
            splits = {s: [json.loads(line) for line in (output / f'{s}.jsonl').read_text().splitlines()]
                      for s in ('train', 'validation', 'test')}
            self.assertEqual([len(splits[s]) for s in splits], [2, 1, 1])
            self.assertEqual(splits['validation'][0]['positive_target_item'], mappings['item_to_id']['Movie:45'])
            self.assertEqual(splits['test'][0]['positive_target_item'], mappings['item_to_id']['Movie:46'])
            for rows in splits.values():
                for row in rows:
                    self.assertLess(row['maximum_source_timestamp'], row['timestamp'])
                    self.assertNotIn(row['positive_target_item'], row['target_items'])
            texts = (output / 'item_texts.jsonl').read_text()
            self.assertIn('TRAIN_TAG', texts)
            for forbidden in ('FUTURE_LABEL', 'TARGET_LABEL_MUST_NOT_ENTER_TEXT', 'FORBIDDEN_RATING', 'FORBIDDEN_TAG'):
                self.assertNotIn(forbidden, texts)
            protocol = json.loads((output / 'protocol.json').read_text())
            self.assertFalse(protocol['paper_alignment_verified'])
            self.assertTrue(verify_processed(output))

    def test_splits_match_existing_amazon_reference_including_repeats(self):
        from cdr_framework.datasets.amazon_preprocessing import (
            AmazonEvent, PreparedDomainPair, build_temporal_splits)
        from datetime import datetime
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); raw = raw_fixture(root, target_repeat=True); output = root / 'processed'
            convert_archive(raw, output, expected_sha256=sha256(raw))
            mappings = json.loads((output / 'mappings.json').read_text())
            def stamp(day): return (datetime(2020, 1, day)-datetime(1970, 1, 1)).days*86400
            src = [AmazonEvent('800', str(42+i), 'Book', stamp(day))
                   for i, day in enumerate((1, 2, 3, 7, 20))]
            src.append(AmazonEvent('800', '42', 'Book', stamp(4)))
            tgt = [AmazonEvent('800', str(42+i), 'Movie', stamp(day))
                   for i, day in enumerate((5, 6, 8, 9, 10))]
            tgt.append(AmazonEvent('800', '42', 'Movie', stamp(11)))
            prepared = PreparedDomainPair('Book', 'Movie', {'800': 0}, mappings['item_to_id'],
                mappings['id_to_item'], {}, {'800': tuple(sorted(src, key=lambda e:(e.timestamp,e.asin)))},
                {'800': tuple(tgt)}, {})
            reference = build_temporal_splits(prepared)
            for split in ('train', 'validation', 'test'):
                actual = [json.loads(line) for line in (output / f'{split}.jsonl').read_text().splitlines()]
                for got, expected in zip(actual, getattr(reference, split)):
                    for key in ('user_id', 'positive_target_item', 'timestamp'):
                        self.assertEqual(got[key], getattr(expected, key))
                    for key in ('source_items', 'target_items'):
                        self.assertEqual(tuple(got[key]), getattr(expected, key))
                self.assertEqual(len(actual), len(getattr(reference, split)))
            stats = json.loads((output / 'statistics.json').read_text())
            self.assertEqual(stats['test_positives_in_seen'], 1)

    def test_exact_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); raw = raw_fixture(root); output = root / 'processed'
            digest = sha256(raw)
            convert_archive(raw, output, expected_sha256=digest)
            original = (output / 'manifest.json').read_bytes()
            convert_archive(raw, output, expected_sha256=digest)
            self.assertEqual(original, (output / 'manifest.json').read_bytes())
            with (output / 'train.jsonl').open('a') as stream: stream.write('{}\n')
            with self.assertRaisesRegex(RuntimeError, 'hash'):
                convert_archive(raw, output, expected_sha256=digest)


if __name__ == '__main__':
    unittest.main()
