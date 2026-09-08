import unittest

from cdr_framework.datasets.schema import CrossDomainSequence, InteractionRecord, ItemMetadataRecord
from cdr_framework.datasets.splits import chronological_leave_one_out


class DatasetSchemaTest(unittest.TestCase):
    def test_records_cover_amazon_and_douban_fields(self):
        interaction = InteractionRecord(
            user_id="u1",
            item_id="i1",
            domain="Sports",
            timestamp=10,
            rating=5.0,
            review_text="light and waterproof",
            behavior_type="review",
        )
        item = ItemMetadataRecord(
            item_id="i1",
            domain="Sports",
            title="Trail Jacket",
            category=("Sports", "Outdoor"),
            brand_or_creator="BrandA",
            description="Waterproof running jacket",
            image_ref="image://jacket",
            attributes={"color": "blue"},
        )

        self.assertEqual(interaction.domain, "Sports")
        self.assertEqual(item.category, ("Sports", "Outdoor"))
        self.assertEqual(item.attributes["color"], "blue")

    def test_chronological_leave_one_out_sorts_and_splits_each_user(self):
        records = [
            InteractionRecord("u1", "i3", "Movies", 30),
            InteractionRecord("u1", "i1", "Books", 10),
            InteractionRecord("u1", "i2", "Books", 20),
            InteractionRecord("u2", "j1", "Phones", 1),
            InteractionRecord("u2", "j2", "Electronics", 2),
            InteractionRecord("u2", "j3", "Electronics", 3),
        ]

        splits = chronological_leave_one_out(records)

        self.assertEqual([record.item_id for record in splits.train], ["i1", "j1"])
        self.assertEqual([record.item_id for record in splits.validation], ["i2", "j2"])
        self.assertEqual([record.item_id for record in splits.test], ["i3", "j3"])

    def test_cross_domain_sequence_exposes_source_and_target_histories(self):
        sequence = CrossDomainSequence(
            user_id="u1",
            source_domain="Books",
            target_domain="Movies",
            source_items=("b1", "b2"),
            target_items=("m1",),
            positive_target_item="m2",
        )

        self.assertEqual(sequence.all_history(), ("b1", "b2", "m1"))


if __name__ == "__main__":
    unittest.main()
