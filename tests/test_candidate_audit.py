import unittest
import json
from pathlib import Path
import tempfile

from cdr_framework.datasets.gencdr_protocol import Event, build_shared

from experiments.audit_cross_domain_candidates import direction_statistics


class CandidateAuditTests(unittest.TestCase):
    def test_temporal_filter_repeat_labels_and_complete_catalog(self):
        source = {"u": [(10, "s1", 1), (11, "s2", 2), (12, "s3", 3),
                        (13, "s4", 4), (14, "s5", 5)], "source_only": [(1, "x", 1)] * 5}
        target = {"u": [(1, "a", 1), (10, "b", 2), (12, "a", 3),
                        (13, "v", 4), (14, "t", 5)]}
        stats = direction_statistics(source, target)
        self.assertEqual(stats["users"], 1)
        self.assertEqual(stats["train_rows"], 1)
        self.assertEqual(stats["train_positives_in_seen"], 1)
        self.assertEqual(stats["validation_rows"], 1)
        self.assertEqual(stats["test_rows"], 1)
        self.assertEqual(stats["source_items"], 5)
        self.assertEqual(stats["target_items"], 4)

    def test_user_must_have_five_events_in_each_domain(self):
        rows = [(i, str(i), i) for i in range(5)]
        stats = direction_statistics({"u": rows}, {"u": rows[:4]})
        self.assertEqual(stats["users"], 0)
        self.assertEqual(stats["train_rows"], 0)

    def test_metadata_overlap_is_reported(self):
        rows = [(i, str(i), i) for i in range(5)]
        stats = direction_statistics({"u": rows}, {"u": rows})
        self.assertEqual(stats["cross_domain_shared_asins"], 5)

    def test_counts_match_actual_sir_protocol_builder_in_both_directions(self):
        source = {"u": [(10, "a", "1"), (11, "a", "2"), (12, "c", "3"),
                        (15, "v", "4"), (18, "t", "5")]}
        target = {"u": [(1, "z", "1"), (10, "b", "2"), (12, "z", "3"),
                        (17, "v", "4"), (20, "t", "5")]}
        events = [Event(user, domain, item, stamp, key)
                  for domain, groups in (("source", source), ("target", target))
                  for user, rows in groups.items() for stamp, item, key in rows]
        texts = {(event.domain, event.item): event.item for event in events}
        with tempfile.TemporaryDirectory() as temp:
            for left, right, a, b in (("source", "target", source, target),
                                     ("target", "source", target, source)):
                output = Path(temp) / left
                build_shared(events, texts, left, right, output, provenance={"fixture": True})
                actual = json.loads((output / "statistics.json").read_text(encoding="utf-8"))
                for key, value in actual.items():
                    if key in direction_statistics(a, b):
                        self.assertEqual(direction_statistics(a, b)[key], value, key)
