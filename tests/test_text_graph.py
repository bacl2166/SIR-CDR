import json
from pathlib import Path
import tempfile
import unittest

import torch

from cdr_framework.text_graph import SparseDualGraphEncoder, build_training_graphs


class TextGraphTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "train.jsonl"

    def build(self, rows, window=1, num_items=10, target_ids=None):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return build_training_graphs(
            self.path, num_items,
            torch.tensor([5, 6, 7, 8, 9]) if target_ids is None else target_ids,
            cross_window=window,
        )

    @staticmethod
    def row(source, target, timestamp=2, user=1, label=7):
        return dict(user_id=user, timestamp=timestamp, source_items=source,
                    target_items=target, positive_target_item=label)

    @staticmethod
    def edges(graph):
        return dict(zip(map(tuple, graph.indices().t().tolist()), graph.values().tolist()))

    def test_latest_prefix_history_only_and_training_file_only(self):
        rows = [self.row([0, 1, 2], [0, 5, 6]), self.row([3, 4], [8, 9], timestamp=1)]
        heldout = self.path.with_name("test.jsonl")
        heldout.write_text(json.dumps(self.row([3, 4], [7, 8], timestamp=3)), encoding="utf-8")
        graphs = self.build(rows)
        loops = {(i, i) for i in range(1, 10)}
        self.assertEqual(set(graphs), {"shared", "source", "target"})
        self.assertEqual(set(self.edges(graphs["source"])), loops | {(1, 2), (2, 1)})
        self.assertEqual(set(self.edges(graphs["target"])), loops | {(5, 6), (6, 5)})
        self.assertEqual(set(self.edges(graphs["shared"])), loops | {
            (1, 2), (2, 1), (5, 6), (6, 5), (2, 6), (6, 2)})
        for graph in graphs.values():
            self.assertEqual(graph.layout, torch.sparse_coo)
            self.assertTrue(graph.is_coalesced())
            self.assertEqual(graph.shape, (10, 10))
            self.assertTrue(torch.isfinite(graph.values()).all())

    def test_exact_symmetric_normalization_and_binary_union(self):
        graphs = self.build([self.row([1, 2, 1, 2], [5, 6]),
                             self.row([1, 2], [5, 6], user=2)])
        edges = self.edges(graphs["shared"])
        degrees = {1: 2, 2: 3, 5: 2, 6: 3}
        for (i, j), value in edges.items():
            self.assertAlmostEqual(value, (degrees.get(i, 1) * degrees.get(j, 1)) ** -0.5, places=6)
            self.assertEqual(value, edges[j, i])

    def test_cross_window_is_cartesian_history_suffix(self):
        graph = self.build([self.row([0, 1, 2, 3, 0], [0, 5, 6, 8, 0])], window=2)["shared"]
        cross = {(i, j) for i, j in self.edges(graph) if i < 5 <= j}
        self.assertEqual(cross, {(2, 6), (2, 8), (3, 6), (3, 8)})
        disabled = self.build([self.row([1, 2], [5, 6])], window=0)["shared"]
        self.assertFalse(any(i < 5 <= j for i, j in self.edges(disabled)))

    def test_ties_last_row_wins_and_previous_labels_can_be_history(self):
        graphs = self.build([self.row([3, 4], [5], label=6), self.row([1, 2], [5, 6])])
        self.assertIn((5, 6), self.edges(graphs["target"]))
        self.assertNotIn((3, 4), self.edges(graphs["source"]))

    def test_empty_training_and_large_catalog_stay_sparse(self):
        for count in (1, 100_000):
            graphs = self.build([], num_items=count, target_ids=torch.empty(0, dtype=torch.long))
            for graph in graphs.values():
                self.assertEqual(graph._nnz(), count - 1)
                self.assertTrue(torch.equal(graph.values(), torch.ones(count - 1)))

    def test_invalid_ids_domains_and_arguments(self):
        for row in [self.row([-1], []), self.row([10], []), self.row([True], []),
                    self.row([1.0], []), self.row([5], []), self.row([], [1]),
                    self.row([], [], label=0), self.row([], [], label=1),
                    self.row([], [], timestamp=float("nan"))]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.build([row])
        for ids in [torch.tensor([0]), torch.tensor([10]), torch.tensor([5.0]),
                    torch.tensor([[5]]), torch.tensor([True])]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.build([], target_ids=ids)
        for window in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                self.build([], window=window)
        with self.assertRaises(ValueError):
            self.build([], num_items=0)

    def test_encoder_residual_mean_projections_and_gradients(self):
        graphs = self.build([self.row([1, 2], [5, 6])])
        encoder = SparseDualGraphEncoder(4, layers=2).double()
        self.assertFalse(any(isinstance(m, torch.nn.Embedding) for m in encoder.modules()))
        self.assertEqual(set(encoder.projections), set(graphs))
        with torch.no_grad():
            for projection in encoder.projections.values():
                projection.weight.copy_(torch.eye(4))
        vectors = torch.randn(10, 4, dtype=torch.float64, requires_grad=True)
        outputs = encoder(vectors, graphs)
        self.assertEqual(len(outputs), 3)
        for name, output in zip(("shared", "source", "target"), outputs):
            graph = graphs[name].to(dtype=vectors.dtype)
            first = torch.sparse.mm(graph, vectors)
            expected = (vectors + first + torch.sparse.mm(graph, first)) / 3
            torch.testing.assert_close(output, expected)
            self.assertTrue(torch.isfinite(output).all())
        sum(output.square().sum() for output in outputs).backward()
        self.assertTrue(torch.isfinite(vectors.grad).all())
        self.assertGreater(vectors.grad.abs().sum().item(), 0)
        for parameter in encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(len({id(p.weight) for p in encoder.projections.values()}), 3)

    def test_zero_layers_and_invalid_encoder_inputs(self):
        graphs = self.build([])
        encoder = SparseDualGraphEncoder(3, layers=0)
        vectors = torch.randn(10, 3)
        for name, output in zip(("shared", "source", "target"), encoder(vectors, graphs)):
            torch.testing.assert_close(output, encoder.projections[name](vectors))
        for dim, layers in [(0, 2), (3, -1), (3, 1.5)]:
            with self.assertRaises(ValueError):
                SparseDualGraphEncoder(dim, layers)
        with self.assertRaises(ValueError):
            encoder(torch.randn(10, 2), graphs)
        with self.assertRaises(ValueError):
            encoder(vectors, {**graphs, "shared": torch.eye(10)})


if __name__ == "__main__":
    unittest.main()
