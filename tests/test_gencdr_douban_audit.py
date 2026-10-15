import io
import unittest
from pathlib import Path
import tempfile

from experiments.prepare_gencdr_douban import audit_reviews, PAPER_COUNTS, cleanup_legacy_assets
from experiments.run_gencdr_benchmarks import _selected_specs, BENCHMARKS


class GenCDRDoubanAuditTests(unittest.TestCase):
    def test_only_douban_can_be_the_third_task(self):
        self.assertEqual([s.name for s in _selected_specs(None)],
                         ['sports_to_clothing', 'phones_to_electronics'])
        self.assertEqual(tuple(BENCHMARKS),
                         ('sports_to_clothing', 'phones_to_electronics', 'douban_book_movie'))

    def test_dates_original_ids_and_unreachable_repeat_are_audited(self):
        text = ('user_id\tbook_id\ttime\n'
                '0\t0\t2020-01-03\n0\t2\t2020-01-01\n'
                '0\t0\t2020-01-02\n1\t9\tbad-date\n')
        stats, users = audit_reviews(io.StringIO(text), 'book_id', min_interactions=3)
        self.assertEqual(users, {'0'})
        self.assertEqual(stats['raw']['interactions'], 4)
        self.assertEqual(stats['filtered'], {'users': 1, 'items': 2, 'interactions': 3})
        self.assertEqual(stats['invalid_dates'], 1)
        self.assertEqual(stats['holdout_positive_in_history']['test'], 1)
        self.assertEqual(stats['holdout_positive_in_history']['validation'], 0)
        self.assertFalse(stats['paper_alignment_verified'])

    def test_required_schema_and_paper_reference(self):
        with self.assertRaisesRegex(ValueError, 'Missing'):
            audit_reviews(io.StringIO('user_id\tbook_id\n1\t2\n'), 'book_id')
        self.assertEqual(PAPER_COUNTS['Books']['users'], 1713)
        self.assertEqual(PAPER_COUNTS['Movies']['items'], 20964)

    def test_cleanup_removes_only_legacy_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('gencdsr_book_movie', 'douban_book_movie', 'phones_to_electronics'):
                path = root / 'data/processed' / name
                path.mkdir(parents=True)
                (path / 'keep.json').write_text('{}')
            self.assertEqual(cleanup_legacy_assets(root), ['data/processed/gencdsr_book_movie'])
            self.assertTrue((root / 'data/processed/douban_book_movie/keep.json').exists())
            self.assertTrue((root / 'data/processed/phones_to_electronics/keep.json').exists())


if __name__ == '__main__':
    unittest.main()
