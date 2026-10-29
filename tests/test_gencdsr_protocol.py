import gzip
import json
from pathlib import Path
import tempfile
import unittest

from cdr_framework.datasets import gencdr_protocol as shared
try:
    from cdr_framework.datasets import gencdsr_protocol as protocol
except ImportError:
    protocol = None


class GenCDSRProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(protocol, 'GenCDSR protocol implementation is required')

    def fixture(self):
        source, target = protocol.DOMAINS[protocol.PAIRS[0]]
        events = []
        def add(user, domain, sequence):
            for i, item in enumerate(sequence):
                events.append(shared.Event(user, domain, item, 2*i+(1 if domain == source else 2), str(i)))
        for user in ('a', 'b', 'c'):
            for domain in (source, target): add(user, domain, ['core']*6)
        add('x', source, ['rare']*2+['core']*3)
        add('x', target, ['bridge']*3+['core']*2)
        add('y', source, ['core']*5)
        add('y', target, ['bridge']*2+['core']*3)
        add('outsider', source, ['rare']*5)
        return events, {(e.domain, e.item): 'static '+e.item for e in events}, source, target

    def build(self, output, events=None):
        raw, items, source, target = self.fixture()
        return protocol.build_pair(raw if events is None else events, items, source, target,
                                   output, provenance={'kind': 'synthetic'})

    def read(self, root, name):
        path = root / name
        if name.endswith('.jsonl'):
            return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        return json.loads(path.read_text(encoding='utf-8'))

    def resign(self, root, name, value):
        path = root / name
        path.write_text(('\n'.join(json.dumps(x) for x in value)+'\n') if name.endswith('.jsonl')
                        else json.dumps(value), encoding='utf-8')
        receipt = self.read(root, 'shared_manifest.json')
        receipt['files'][name] = shared.sha(path)
        shared.write(root / 'shared_manifest.json', receipt)

    def test_global_item_support_from_non_common_users_is_not_discarded_before_filtering(self):
        events, _, source, target = self.fixture()
        filtered, trace = protocol._filter(events, source, target)
        self.assertEqual({e.user for e in filtered}, {'a', 'b', 'c', 'x', 'y'})
        self.assertEqual(sum(e.domain == source and e.item == 'rare' for e in filtered), 2)
        self.assertNotIn('outsider', {e.user for e in filtered})

    def test_domain_user_pruning_does_not_delete_the_users_other_domain_support(self):
        events, _, source, target = self.fixture()
        events = [e for e in events if e.user != 'outsider']
        filtered, trace = protocol._filter(events, source, target)
        self.assertEqual({e.user for e in filtered}, {'a', 'b', 'c', 'y'})
        self.assertEqual(sum(e.domain == target and e.item == 'bridge' for e in filtered), 2)

    def test_full_domain_fixedpoint_precedes_intersection_and_keeps_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.build(Path(tmp)/'new')
            self.assertEqual(self.read(root, 'mappings.json')['id_to_user'], ['a', 'b', 'c', 'x', 'y'])
            events = self.read(root, 'events.jsonl')
            self.assertEqual(len(events), 56)
            self.assertEqual({e['item'] for e in events}, {'core', 'rare', 'bridge'})
            provenance = self.read(root, 'protocol.json')['identity']['provenance']
            trace = provenance['filter_audit']
            self.assertEqual(trace['initial_common']['users'], 5)
            self.assertEqual(len(trace['iterations']), 1)
            self.assertEqual(trace['filtered_domains']['users'], 6)
            self.assertEqual(trace['final']['users'], 5)
            self.assertEqual(provenance['converter_sha256'], shared.sha(protocol.__file__))
            report = protocol.audit(root)
            self.assertTrue(report['complete'])
            self.assertEqual(report['users'], 5)
            self.assertFalse(report['raw_filter_replayed'])

    def test_causal_splits_and_reverse_symmetry(self):
        events, items, source, target = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            forward = self.build(Path(tmp)/'f')
            reverse = protocol.build_pair(reversed(events), items, target, source, Path(tmp)/'r', provenance={'kind': 'synthetic'})
            self.assertTrue(protocol.audit_pair_roots(forward, reverse)['complete'])
            rows = self.read(forward, 'train.jsonl')
            self.assertEqual(len(rows), 13)
            self.assertEqual({r['target_position'] for r in rows}, {1, 2, 3})
            for row in rows:
                self.assertLess(row['maximum_source_timestamp'], row['timestamp'])
                self.assertLessEqual(len(row['source_items']), 4)
                self.assertTrue(row['target_items'])
            self.assertFalse(self.read(forward, 'protocol.json')['paper_alignment_verified'])

    def test_rehashed_malformed_rows_labels_positions_and_exclusions_are_rejected(self):
        mutations = [('train.jsonl', 'source_items', []), ('train.jsonl', 'target_items', []),
                     ('test.jsonl', 'positive_target_item', 0), ('validation.jsonl', 'target_position', 0),
                     ('test.jsonl', 'timestamp', 999)]
        with tempfile.TemporaryDirectory() as tmp:
            for n, (name, key, value) in enumerate(mutations):
                root = self.build(Path(tmp)/str(n))
                rows = self.read(root, name); rows[0][key] = value
                self.resign(root, name, rows)
                with self.assertRaises((ValueError, RuntimeError)):
                    protocol.audit(root)
            root = self.build(Path(tmp)/'excluded')
            self.resign(root, 'excluded_samples.jsonl', [{'user': 'a', 'split': 'test'}])
            with self.assertRaises((ValueError, RuntimeError)): protocol.audit(root)

    def test_missing_metadata_has_id_fallback_and_default_shared_schema_is_unchanged(self):
        events, _, source, target = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = protocol.build_pair(events, {}, source, target, Path(tmp)/'new', provenance={})
            self.assertTrue(all(r['text'].strip() for r in self.read(root, 'item_texts.jsonl')))
            old = shared.build_shared(events[:12], {(e.domain, e.item): 'text' for e in events},
                                      source, target, Path(tmp)/'old', provenance={})
            self.assertEqual(self.read(old, 'protocol.json')['schema'], shared.SCHEMA)

    def test_convert_reads_raw_checkout_and_writes_separate_output_namespace(self):
        events, _, source, target = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/'raw'; destination = Path(tmp)/'destination'
            folder = raw/'data/raw/amazon2014'; folder.mkdir(parents=True)
            for domain in (source, target):
                with gzip.open(folder/f'reviews_{domain}_5.json.gz', 'wt', encoding='utf-8') as stream:
                    for event in events:
                        if event.domain == domain:
                            stream.write(json.dumps({'reviewerID': event.user, 'asin': event.item,
                                                    'unixReviewTime': event.timestamp})+'\n')
                with gzip.open(folder/f'meta_{domain}.json.gz', 'wt', encoding='utf-8') as stream:
                    stream.write('{}\n')
            result = protocol.convert(raw, protocol.PAIRS[0], output_root=destination)
            self.assertEqual(result, destination/'data/processed'/protocol.SCHEMA/protocol.PAIRS[0])
            self.assertTrue(protocol.audit(result)['complete'])
            self.assertTrue(protocol.audit(result, raw_root=raw)['raw_filter_replayed'])
            with gzip.open(folder/f'reviews_{source}_5.json.gz', 'at', encoding='utf-8') as stream:
                stream.write(json.dumps({'reviewerID': 'extra', 'asin': 'extra', 'unixReviewTime': 1})+'\n')
            with self.assertRaisesRegex(ValueError, 'raw replay'):
                protocol.audit(result, raw_root=raw)
            self.assertFalse((raw/'data/processed').exists())

    def test_shared_overrides_are_documentation_only_and_bound_to_identity(self):
        events, items, source, target = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            for n, field in enumerate(('population', 'source_visibility', 'schema', 'identity')):
                with self.assertRaisesRegex(ValueError, 'override'):
                    shared.build_shared(events, items, source, target, Path(tmp)/str(n), provenance={},
                                        protocol_overrides={field: 'unsafe'})
            path = shared.build_shared(events, items, source, target, Path(tmp)/'safe', provenance={},
                                       protocol_overrides={'checkpoint_selection': 'NDCG@10'})
            identity = self.read(path, 'shared_manifest.json')['identity']
            self.assertEqual(identity['protocol_overrides'], {'checkpoint_selection': 'NDCG@10'})
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                shared.build_shared(events, items, source, target, path, provenance={},
                                    protocol_overrides={'checkpoint_selection': 'HR@10'})

    def test_empty_source_exclusions_keep_original_label_and_full_population(self):
        events, _, source, target = self.fixture()
        changed = [shared.Event(e.user, e.domain, e.item,
                               e.timestamp+100 if e.user == 'a' and e.domain == source else e.timestamp, e.key)
                   for e in events]
        with tempfile.TemporaryDirectory() as tmp:
            root = self.build(Path(tmp)/'new', changed)
            exclusions = self.read(root, 'excluded_samples.jsonl')
            self.assertEqual(len(exclusions), 5)
            self.assertEqual({r['user'] for r in exclusions}, {'a'})
            self.assertEqual({r['target_event_key'] for r in exclusions if r['split'] == 'test'}, {'5'})
            self.assertEqual(self.read(root, 'mappings.json')['id_to_user'], ['a', 'b', 'c', 'x', 'y'])
            self.assertTrue(protocol.audit(root)['complete'])

    def test_holdout_only_catalog_and_stable_ties_with_distinct_source_items(self):
        source, target = protocol.DOMAINS[protocol.PAIRS[0]]
        events = [shared.Event(user, domain, f'{domain}{i}',
                               2*(i//2)+(1 if domain == source else 2), str(i))
                  for user in ('a', 'b', 'c') for domain in (source, target) for i in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            a = protocol.build_pair(events, {}, source, target, Path(tmp)/'a', provenance={})
            b = protocol.build_pair(reversed(events), {}, source, target, Path(tmp)/'b', provenance={})
            self.assertEqual(self.read(a, 'shared_manifest.json'), self.read(b, 'shared_manifest.json'))
            mapping = self.read(a, 'mappings.json')['item_to_id']
            heldouts = {mapping[f'{source}:{source}{i}'] for i in (4, 5)}
            heldout_targets = {mapping[f'{target}:{target}{i}'] for i in (4, 5)}
            rows = self.read(a, 'train.jsonl')
            self.assertFalse(heldout_targets & {i for r in rows for i in r['target_items']})
            for split in ('train', 'validation', 'test'):
                for row in self.read(a, split+'.jsonl'):
                    self.assertFalse(heldouts & set(row['source_items']))
            self.assertEqual(len(mapping), 12)
            self.assertTrue(protocol.audit(a)['complete'])

    def test_cross_direction_audit_rejects_different_retained_event_sets(self):
        events, items, source, target = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            forward = self.build(Path(tmp)/'f')
            changed = [shared.Event(e.user, e.domain, e.item, e.timestamp+20, e.key) for e in events]
            reverse = protocol.build_pair(changed, items, target, source, Path(tmp)/'r', provenance={})
            self.assertTrue(protocol.audit(reverse)['complete'])
            with self.assertRaisesRegex(ValueError, 'cross-direction'):
                protocol.audit_pair_roots(forward, reverse)

    def test_invalid_raw_event_identity_is_rejected_even_outside_final_population(self):
        events, _, source, target = self.fixture()
        for invalid in (events+[events[-1]], events+[shared.Event('bad', source, 'i', True, 'k')]):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError): self.build(Path(tmp)/'new', invalid)


if __name__ == '__main__': unittest.main()
