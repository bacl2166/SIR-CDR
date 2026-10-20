import json
from pathlib import Path
import tempfile
import unittest

from cdr_framework.datasets.gencdr_protocol import Event, build_shared, verify_shared


class SharedProtocolTests(unittest.TestCase):
    def fixture(self):
        # Last two SOURCE events are held out even when earlier than target labels.
        source = [Event('u', 'S', f's{i}', i, str(i)) for i in range(1, 6)]
        target = [Event('u', 'T', f't{i}', i + 5, str(i)) for i in range(1, 6)]
        items = {('S', f's{i}'): f'static s{i}' for i in range(1, 6)}
        items.update({('T', f't{i}'): f'static t{i}' for i in range(1, 6)})
        return source + target, items

    def test_split_is_prefix_based_and_source_holdouts_never_enter_context(self):
        events, items = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'new'
            build_shared(events, items, 'S', 'T', out, provenance={'kind': 'common-protocol'})
            mapping = json.loads((out / 'mappings.json').read_text())
            rows = {s: [json.loads(l) for l in (out / f'{s}.jsonl').read_text().splitlines()]
                    for s in ('train', 'validation', 'test')}
            self.assertEqual(len(rows['train']), 2)
            self.assertEqual(rows['validation'][0]['positive_target_item'], mapping['item_to_id']['T:t4'])
            self.assertEqual(rows['test'][0]['positive_target_item'], mapping['item_to_id']['T:t5'])
            bridge = json.loads((out / 'gencdr_sequence_splits.jsonl').read_text().splitlines()[0])
            self.assertEqual(bridge['accepted_target_positions'], {'train': [1, 2], 'validation': [3], 'test': [4]})
            for split in rows.values():
                for row in split:
                    self.assertEqual(len(row['source_items']), 3)
                    self.assertLess(row['maximum_source_timestamp'], row['timestamp'])
            self.assertTrue(verify_shared(out))
            (out / 'train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(RuntimeError, 'hash'):
                verify_shared(out)

    def test_exact_ties_are_deterministic_and_duplicates_are_not_silently_removed(self):
        events, items = self.fixture()
        events[-2] = Event('u', 'T', 't3', 9, '4')
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'new'
            build_shared(reversed(events), items, 'S', 'T', out, provenance={'kind': 'common-protocol'})
            stats = json.loads((out / 'statistics.json').read_text())
            self.assertEqual(stats['validation_positives_in_seen'], 1)
            self.assertFalse(json.loads((out / 'protocol.json').read_text())['paper_alignment_verified'])

    def test_future_source_does_not_move_holdout_to_an_easier_event(self):
        events, items = self.fixture()
        events = [Event(e.user, e.domain, e.item, e.timestamp + 100 if e.domain == 'S' else e.timestamp, e.key)
                  for e in events]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'empty'):
                build_shared(events, items, 'S', 'T', Path(folder) / 'new', provenance={'kind': 'common-protocol'})

    def test_unknown_items_or_reused_event_keys_are_rejected(self):
        events, items = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            for changed in (events + [events[0]], events + [Event('u', 'T', 'missing', 20, '99')]):
                with self.assertRaises(ValueError):
                    build_shared(changed, items, 'S', 'T', Path(folder) / 'new', provenance={'kind': 'common-protocol'})


if __name__ == '__main__':
    unittest.main()
