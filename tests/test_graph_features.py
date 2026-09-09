import unittest

import torch

from cdr_framework.data import GraphBatch
from cdr_framework.graphs.builders import build_transition_edges
from cdr_framework.graphs.confidence import GraphEdgeFeatures, edge_feature_tensor
from cdr_framework.modules import SharedSpecificGraphEncoder


class GraphFeatureTest(unittest.TestCase):
    def test_build_transition_edges_counts_adjacent_sequences(self):
        sequences = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long)

        edges, counts = build_transition_edges(sequences, num_items=4)

        self.assertEqual(edges.tolist(), [[0, 1], [1, 2], [1, 3]])
        self.assertEqual(counts.tolist(), [2.0, 1.0, 1.0])

    def test_edge_feature_tensor_keeps_expected_feature_order(self):
        features = [
            GraphEdgeFeatures(edge_type_id=1, semantic_score=0.2, visual_score=0.3, category_score=0.4, co_user_count=2),
            GraphEdgeFeatures(edge_type_id=2, llm_score=0.9, transition_count=3),
        ]

        tensor = edge_feature_tensor(features, device=torch.device("cpu"), dtype=torch.float32)

        self.assertEqual(tensor.shape, (2, GraphEdgeFeatures.width()))
        self.assertAlmostEqual(tensor[0, 1].item(), 0.2)
        self.assertAlmostEqual(tensor[1, 6].item(), 0.9)

    def test_graph_encoder_accepts_optional_edge_features(self):
        torch.manual_seed(3)
        item_tokens = torch.randn(4, 5)
        source_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        target_edges = torch.tensor([[2, 3]], dtype=torch.long)
        cross_edges = torch.tensor([[0, 3]], dtype=torch.long)
        graph = GraphBatch(
            source_edges=source_edges,
            target_edges=target_edges,
            cross_edges=cross_edges,
            num_items=4,
            source_edge_features=torch.ones((2, GraphEdgeFeatures.width())),
            target_edge_features=torch.ones((1, GraphEdgeFeatures.width())),
            cross_edge_features=torch.ones((1, GraphEdgeFeatures.width())),
        )
        encoder = SharedSpecificGraphEncoder(hidden_dim=5, edge_feature_dim=GraphEdgeFeatures.width())

        cd_graph, source_graph, target_graph = encoder(item_tokens, graph)

        self.assertEqual(cd_graph.shape, (4, 5))
        self.assertEqual(source_graph.shape, (4, 5))
        self.assertEqual(target_graph.shape, (4, 5))


if __name__ == "__main__":
    unittest.main()
