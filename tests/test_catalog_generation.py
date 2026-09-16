import inspect
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from cdr_framework.catalog_generation import CatalogTrie, constrained_generate
from cdr_framework.modules import AutoregressiveSemanticDecoder


class TableDecoder(nn.Module):
    bos_id = 4
    eos_id = 5

    def __init__(self, table=None):
        super().__init__()
        self.table = table or {}
        self.calls = []

    def forward(self, prefix, input_tokens):
        assert not torch.is_grad_enabled()
        self.calls.append(input_tokens.tolist())
        rows = [self.table.get(tuple(row[1:]), [0.0] * 6)
                for row in input_tokens.tolist()]
        return torch.tensor(rows, device=prefix.device).unsqueeze(1).expand(
            -1, input_tokens.shape[1], -1
        )


class CatalogGenerationTests(unittest.TestCase):
    def setUp(self):
        self.semantic = torch.tensor([[-1, -1], [3, 3], [0, 1], [2, 0],
                                      [0, 1], [0, 2]])
        self.ids = torch.tensor([5, 0, 4, 2, 3, 4])
        self.trie = CatalogTrie(self.ids, self.semantic)
        self.prefix = torch.zeros(1, 2, 3, requires_grad=True)
        self.query = torch.tensor([[1.0, 0.0]], requires_grad=True)
        self.vectors = torch.tensor([[0., 0.], [1., 0.], [1., 1.],
                                     [-1., 0.], [10., 0.], [1., 0.]])

    def generate(self, decoder=None, **kwargs):
        args = dict(decoder=decoder or TableDecoder(), prefix=self.prefix,
                    trie=self.trie, beam_size=10, top_k=6,
                    seen_items=[set()], query=self.query,
                    item_vectors=self.vectors)
        args.update(kwargs)
        return constrained_generate(**args)

    def test_global_rows_legal_tokens_collisions_and_dedup(self):
        self.assertEqual(self.trie.next_codes(()), (0, 2))
        self.assertEqual(self.trie.next_codes((0,)), (1, 2))
        self.assertEqual(self.trie.terminal_items((0, 1)), (2, 4))
        self.assertEqual(self.trie.next_codes((3,)), ())
        decoder = TableDecoder({(): [0., 0., 0., 100., 90., 80.]})
        self.assertEqual(self.generate(decoder).tolist(), [[4, 2, 5, 3, 0, 0]])
        legal_prefixes = {(), (0,), (2,), (0, 1), (0, 2), (2, 0)}
        for batch in decoder.calls:
            for row in batch:
                self.assertEqual(row[0], decoder.bos_id)
                self.assertIn(tuple(row[1:]), legal_prefixes)
        self.assertEqual([len(batch) for batch in decoder.calls], [1, 2, 3])

    def test_seen_masks_all_items_and_pads_without_fallback(self):
        result = self.generate(prefix=self.prefix.expand(2, -1, -1),
                               query=self.query.expand(2, -1),
                               seen_items=[{4, 5}, {2, 3, 4, 5}])
        self.assertEqual(result.tolist(), [[2, 3, 0, 0, 0, 0], [0] * 6])
        self.assertEqual(result.dtype, torch.long)
        self.assertEqual(result.device, self.prefix.device)
        self.assertFalse(result.requires_grad)
        self.assertEqual(self.generate(beam_size=1, seen_items=[{2, 4}]).tolist(),
                         [[0] * 6])

    def test_eos_is_scored_before_final_beam_pruning(self):
        trie = CatalogTrie(torch.tensor([1, 2]), torch.tensor([[-1], [0], [1]]))
        decoder = TableDecoder({(): [3., 2., -10., -10., -10., -10.],
                                (0,): [10., 0., 0., 0., 0., -10.],
                                (1,): [0., 0., 0., 0., 0., 10.]})
        self.assertEqual(self.generate(decoder, trie=trie, beam_size=1,
                                       top_k=2, item_vectors=self.vectors[:3]).tolist(), [[2, 0]])

    def test_offset_codes_and_zero_padding_row_parent_api(self):
        tokens = torch.tensor([[0, 0], [0, 3], [1, 2], [0, 3]])
        target_ids = torch.tensor([0, 3, 1, 2])
        trie = CatalogTrie(target_ids.cpu(), tokens.cpu())
        decoder = AutoregressiveSemanticDecoder(2 * 2, 3, 2).eval()
        result = constrained_generate(
            decoder, self.prefix, trie, beam_size=4, top_k=4,
            seen_items=[{1}], query=self.query, item_vectors=self.vectors[:4]
        )
        self.assertEqual(set(result[0, :2].tolist()), {2, 3})
        self.assertEqual(result[0, 2:].tolist(), [0, 0])
        self.assertEqual(trie.terminal_items((0, 3)), (1, 3))

    def test_full_vocab_probabilities_are_not_renormalized(self):
        decoder = TableDecoder({(): [2., -10., 1., -10., -10., -10.],
                                (0,): [0., 0., 0., 12., 0., 0.],
                                (2,): [1., 0., 0., 0., 0., 0.]})
        result = self.generate(decoder)
        self.assertEqual(result[0, 0].item(), 3)

    def test_cosine_only_breaks_collisions_and_item_id_breaks_cosine_ties(self):
        vectors = self.vectors.clone()
        vectors[2] = torch.tensor([2., 0.])
        vectors[4] = torch.tensor([100., 0.])
        self.assertEqual(self.generate(item_vectors=vectors).tolist(),
                         [[2, 4, 5, 3, 0, 0]])
        # Equal sequence scores keep sequences together, independent of cosine.
        reverse = self.generate(query=-self.query)
        self.assertEqual(reverse.tolist(), [[2, 4, 5, 3, 0, 0]])

    def test_label_free_api(self):
        self.assertEqual(list(inspect.signature(constrained_generate).parameters),
                         ['decoder', 'prefix', 'trie', 'beam_size', 'top_k',
                          'seen_items', 'query', 'item_vectors'])
        self.assertEqual(self.generate().tolist(), self.generate().tolist())

    def test_empty_catalog_and_empty_batch(self):
        for ids in (torch.tensor([], dtype=torch.long), torch.tensor([0])):
            decoder = TableDecoder()
            result = self.generate(decoder, trie=CatalogTrie(ids, self.semantic))
            self.assertEqual(result.tolist(), [[0] * 6])
            self.assertEqual(decoder.calls, [])
        result = self.generate(prefix=self.prefix[:0], query=self.query[:0], seen_items=[])
        self.assertEqual(tuple(result.shape), (0, 6))

    def test_exhaustive_parity_with_real_decoder(self):
        torch.manual_seed(17)
        decoder = AutoregressiveSemanticDecoder(4, 3, 2).eval()
        prefix = torch.randn(2, 2, 3)
        query = torch.tensor([[1., 0.], [-1., 0.]])
        seen = [{4}, {3, 5}]
        sequences = {(0, 1): [2, 4], (0, 2): [5], (2, 0): [3]}
        expected = []
        with torch.no_grad():
            for user in range(2):
                scored = []
                for sequence, items in sequences.items():
                    tokens = torch.tensor([[decoder.bos_id, *sequence]])
                    log_probs = decoder(prefix[user:user + 1], tokens).log_softmax(-1)
                    targets = torch.tensor([*sequence, decoder.eos_id])
                    score = log_probs[0, torch.arange(3), targets].sum().item()
                    ranked = sorted((item for item in items if item not in seen[user]),
                                    key=lambda item: (-F.cosine_similarity(
                                        query[user:user + 1], self.vectors[item:item + 1]
                                    ).item(), item))
                    scored.append((score, sequence, ranked))
                ordered = [item for _, _, items in sorted(scored, key=lambda x: (-x[0], x[1]))
                           for item in items]
                expected.append(ordered + [0] * (6 - len(ordered)))
        result = self.generate(decoder, prefix=prefix, query=query, seen_items=seen)
        self.assertEqual(result.tolist(), expected)

    def test_validation(self):
        for name in ('beam_size', 'top_k'):
            for value in (0, -1, 1.5, True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.generate(**{name: value})
        invalid = [dict(prefix=torch.zeros(1, 3)), dict(query=torch.zeros(2, 2)),
                   dict(query=torch.zeros(1, 3)), dict(item_vectors=torch.zeros(2)),
                   dict(item_vectors=torch.zeros(3, 2)), dict(seen_items=[])]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.generate(**kwargs)
        for ids, semantic in [(self.ids[:, None], self.semantic),
                              (self.ids, self.semantic[0]),
                              (torch.tensor([-1]), self.semantic),
                              (torch.tensor([6]), self.semantic),
                              (torch.tensor([1.]), self.semantic),
                              (self.ids, self.semantic.float()),
                              (torch.tensor([1]), torch.tensor([[-1], [-1]])),
                              (torch.tensor([1]), torch.empty(2, 0, dtype=torch.long))]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                CatalogTrie(ids, semantic)
        bad = CatalogTrie(torch.tensor([1]), torch.tensor([[-1], [4]]))
        with self.assertRaises(ValueError):
            self.generate(trie=bad)


if __name__ == '__main__':
    unittest.main()
