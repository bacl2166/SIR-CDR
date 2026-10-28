"""New reconstructed rows flow through the unchanged real SIR/Qwen modules."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cdr_framework.datasets.gencdr_protocol import Event
from cdr_framework.datasets import gencdsr_protocol as protocol
from cdr_framework.llm_config import LLMCDRConfig
from cdr_framework import llm_training
from cdr_framework.text_data import load_text_rows, collate_text_rows
from test_llm_training import offline_backbone


def input_events():
    source, target = protocol.DOMAINS['sports_to_clothing']
    events, texts = [], {}
    for domain, offset in ((source, 0), (target, 1)):
        for index in range(6):
            item = f'item{index}'
            texts[domain, item] = f'Static product {domain} {item}'
            for user in range(4):
                events.append(Event(str(user), domain, item, 10+20*index+offset,
                                    f'{user}-{index}'))
    return events, texts


class ProtocolSIRIntegrationTests(unittest.TestCase):
    def test_freeze_rejects_different_but_individually_valid_direction_snapshots(self):
        from experiments import run_gencdsr_sports as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, texts = input_events()
            for index, pair in enumerate(protocol.PAIRS):
                source, target = protocol.DOMAINS[pair]
                altered = [Event(e.user, e.domain, e.item, e.timestamp+index*100, e.key)
                           for e in events]
                protocol.build_pair(altered, texts, source, target,
                    runner.paths(root, pair)['processed'], provenance={'fixture': True})
                protocol.audit(runner.paths(root, pair)['processed'])
            with patch.object(runner, 'binding', side_effect=lambda root, pair, seed, study:
                              {'pair': pair, 'seed': seed, 'complete': True}):
                with self.assertRaisesRegex(ValueError, 'cross-direction'):
                    runner.freeze(root, 'study')
            self.assertFalse(runner.freeze_path(root, 'study').exists())

    def test_confirmation_waits_for_both_direction_pilots(self):
        from contextlib import contextmanager
        from types import SimpleNamespace
        from experiments import run_gencdsr_sports as runner
        @contextmanager
        def gpu(*args, **kwargs):
            yield torch.device('cpu'), {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = runner.paths(root, protocol.PAIRS[0], 43, 'study')['job']
            runner.immutable(job / 'smoke.json', {'passed': True})
            with patch.object(runner, 'gpu_session', side_effect=gpu), \
                    patch.object(runner, 'verify_runtime', return_value=SimpleNamespace()), \
                    patch.object(runner, 'binding', side_effect=[{'pilot': True}, FileNotFoundError('reverse pilot pending')]) as bind, \
                    patch('cdr_framework.llm_training.train') as train:
                with self.assertRaisesRegex(FileNotFoundError, 'reverse pilot pending'):
                    runner.workflow(root, root/'raw', 'study', protocol.PAIRS[0], 43, 'train', 'GPU-placeholder')
                self.assertEqual(bind.call_count, 2)
                train.assert_not_called()

    def test_both_directions_train_original_modules_and_keep_holdouts_out_of_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, texts = input_events()
            outputs = []
            for pair in protocol.PAIRS:
                source, target = protocol.DOMAINS[pair]
                processed = root / pair
                protocol.build_pair(events, texts, source, target, processed,
                                    provenance={'fixture': 'synthetic-static'})
                protocol.audit(processed)
                outputs.append(processed)
                mapping = json.loads((processed / 'mappings.json').read_text(encoding='utf-8'))
                n = len(mapping['id_to_item'])
                token = root / (pair+'_tokenizer')
                token.mkdir()
                with torch.random.fork_rng():
                    torch.manual_seed(133)
                    latent = torch.randn(n, 4)
                    latent[0] = 0
                    ids = torch.randint(0, 8, (n, 2))
                    ids[0] = -1
                    torch.save(latent, token / 'item_latents.pt')
                    torch.save(ids, token / 'semantic_ids.pt')
                    torch.save({'schema_version': 2, 'codebooks': torch.randn(2, 8, 4)},
                               token / 'tokenizer.pt')
                (token / 'quality_report.json').write_text('{"passed":true}', encoding='utf-8')
                config = LLMCDRConfig(processed_dir=processed, tokenizer_dir=token,
                    output_dir=root / (pair+'_model'), hidden_dim=4, llm_dtype='float32',
                    llm_revision='a'*40, lora_rank=2, lora_alpha=4, batch_size=2,
                    evaluation_batch_size=2, gradient_accumulation_steps=2,
                    reasoning_steps=2, feedback_steps=1, rerank_candidates=6,
                    decode_chunk_size=4, max_sequence_length=50, dropout=0.,
                    top_ks=(5, 10), inference_mode='retrieval')
                with patch('cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained', offline_backbone):
                    model, catalog = llm_training.build_model(config, torch.device('cpu'))
                rows = load_text_rows(processed / 'train.jsonl', 50, n, catalog.target_item_ids)
                batch, seen = collate_text_rows(rows[:2])
                losses = model(batch)
                self.assertTrue(torch.isfinite(losses['total']))
                losses['total'].backward()
                for name in ('graph_encoder', 'prototype', 'cd_injector', 'sp_injector',
                             'codebook_summary', 'reasoner', 'cpf', 'decoder'):
                    module = getattr(model, name)
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in module.parameters()), name)
                self.assertTrue(any('lora_' in name and p.grad is not None
                                    for name, p in model.named_parameters()))
                heldouts = {index for index, record in enumerate(mapping['id_to_item'])
                            if record and record['original_id'] in ('item4', 'item5')}
                for name in ('shared', 'source', 'target'):
                    edges = getattr(model, 'graph_'+name).coalesce().indices()
                    off_diagonal = edges[:, edges[0] != edges[1]]
                    self.assertFalse(set(off_diagonal.flatten().tolist()) & heldouts)
                model.eval()
                validation = load_text_rows(processed / 'validation.jsonl', 50, n, catalog.target_item_ids)
                result = llm_training.evaluate(model, validation, config, torch.device('cpu'))
                self.assertEqual(result['examples'], 4)
                for cutoff in (5, 10):
                    self.assertLessEqual(result[f'NDCG@{cutoff}'], result[f'HR@{cutoff}']+1e-12)
            protocol.audit_pair_roots(*outputs)


if __name__ == '__main__':
    unittest.main()
