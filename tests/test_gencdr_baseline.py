import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch


class GenCDRAssetsTests(unittest.TestCase):
    def test_existing_rows_including_repeat_positives_are_read_verbatim(self):
        from cdr_framework.baselines.gencdr.assets import load_rows
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.jsonl'
            row = dict(user_id=7, source_items=[1, 2], target_items=[3], positive_target_item=4)
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            before = path.read_bytes()
            self.assertEqual(load_rows(path, {1, 2}, {3, 4}), [row])
            self.assertEqual(path.read_bytes(), before)
            row['positive_target_item'] = 3
            path.write_text(json.dumps(row), encoding='utf-8')
            self.assertEqual(load_rows(path, {1, 2}, {3, 4}), [row])

    def test_source_completion_needs_receipts_and_hash_bound_reports(self):
        from cdr_framework.baselines.gencdr.protocol import sir_completion
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(sir_completion(Path(tmp), 'source'))
            out = Path(tmp) / 'artifacts/amazon_recommendation/source'
            out.mkdir(parents=True)
            (out / 'results.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'freeze|incomplete'):
                sir_completion(Path(tmp), 'source')


class GenCDRModelTests(unittest.TestCase):
    def test_quantizer_gradients_and_frozen_codebooks_during_domain_adaptation(self):
        from cdr_framework.baselines.gencdr.tokenizer import AdaptiveRQVAE
        torch.manual_seed(2)
        model = AdaptiveRQVAE(6, hidden=(12, 8), latent=4, levels=2, size=4, rank=2)
        x, domains = torch.randn(5, 6), torch.tensor([0, 1, 0, 1, 0])
        model.loss(x, domains, 'universal').backward()
        self.assertGreater(model.encoder.blocks[0][0].weight.grad.abs().sum().item(), 0)
        model.set_phase('domain')
        before = copy.deepcopy(model.quantizer.state_dict())
        model.zero_grad(set_to_none=True)
        model.loss(x, domains, 'domain').backward()
        self.assertIsNone(model.encoder.blocks[0][0].weight.grad)
        self.assertTrue(all(torch.equal(v, model.quantizer.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapters.parameters()))
        codes = model.codes(x, domains)
        self.assertEqual(codes.shape, (5, 2))
        self.assertTrue(((codes >= 0) & (codes < 4)).all())

    def test_beam_search_is_label_free_masks_seen_and_handles_collisions(self):
        from cdr_framework.baselines.gencdr.decoding import DomainTrie, beam_rank, ranking_metrics
        codes = torch.tensor([[0, 0], [0, 0], [0, 1], [1, 0], [1, 0]])
        trie = DomainTrie(codes, [1, 2, 3, 4])
        ranked = beam_rank(lambda prefixes: torch.zeros(len(prefixes), 2), trie, {1}, beam=10, top_k=10)
        self.assertNotIn(1, ranked)
        self.assertEqual(set(ranked), {2, 3, 4})
        self.assertEqual(len(ranked), len(set(ranked)))
        self.assertEqual(ranking_metrics([ranked], [99])['HR@10'], 0)
        self.assertAlmostEqual(ranking_metrics([[8, 9]], [9])['NDCG@5'], 1 / __import__('math').log2(3))

    def test_lora_experts_and_domain_phase_freeze_without_backbone_training(self):
        from transformers import Qwen3Config, Qwen3Model
        from cdr_framework.baselines.gencdr.model import GenCDRDecoder
        torch.manual_seed(3)
        backbone = Qwen3Model(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8))
        model = GenCDRDecoder(backbone, levels=2, size=4, experts=2, rank=2, alpha=4)
        tokens, mask = torch.tensor([[1, 2, 4, 5], [1, 3, 6, 0]]), torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        labels = torch.tensor([[0, 1], [1, 0]])
        model.loss(tokens, mask, labels, 'universal').backward()
        self.assertTrue(all(p.grad is None for n, p in model.backbone.named_parameters() if '.base.' in n))
        model.set_phase('domain')
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
        optimizer.zero_grad()
        model.loss(tokens, mask, labels, 'domain').backward()
        optimizer.step()
        for n, p in model.named_parameters():
            if n in frozen:
                self.assertTrue(torch.equal(p, frozen[n]), n)
        self.assertEqual(model.next_log_probs(tokens, mask, [(), ()], 'domain').shape, (2, 4))


if __name__ == '__main__':
    unittest.main()
