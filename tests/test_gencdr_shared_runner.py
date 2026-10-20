import json
from pathlib import Path
import tempfile
import unittest

from cdr_framework.datasets.gencdr_inputs import load_bundle
from experiments.run_gencdr_shared import paths, model_config, validate_asset_chain


class SharedRunnerTests(unittest.TestCase):
    def test_new_configs_are_isolated_and_select_recall_without_changing_core(self):
        for pair in ('sports_to_clothing', 'phones_to_electronics', 'douban_book_movie', 'douban_movie_book'):
            root = Path('/workspace').resolve()
            cfg = model_config(root, pair, 43)
            self.assertIn('gencdr-shared-loo-v1', str(cfg.processed_dir))
            self.assertIn('gencdr-shared-loo-v1', str(cfg.tokenizer_dir))
            self.assertIn('gencdr-shared-loo-v1', str(cfg.output_dir))
            self.assertEqual(cfg.selection_metric, 'HR@10')
            self.assertEqual(cfg.seed, 43)
            self.assertEqual(cfg.batch_size * cfg.gradient_accumulation_steps, 16)

    def test_external_sequence_bridge_preserves_boundaries_and_rejects_wrong_order(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p/'protocol.json').write_text(json.dumps({'text_policy': 'static-or-id-only', 'source_description': 'fixture'}))
            def lines(name, rows):
                (p/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            lines('catalog.jsonl', [{'domain': d, 'original_id': str(i), 'text': 'ID '+str(i)}
                                   for d in ('S', 'T') for i in range(5)])
            lines('events.jsonl', [{'user': 'u', 'domain': d, 'item': str(i), 'timestamp': i, 'key': str(i)}
                                  for d in ('S', 'T') for i in range(5)])
            rows = [{'user': 'u', 'domain': d, 'train': ['0','1','2'], 'val': ['0','1','2','3'],
                     'test': ['0','1','2','3','4']} for d in ('S', 'T')]
            lines('sequence_splits.jsonl', rows)
            events, items, _ = load_bundle(p, 'S', 'T')
            self.assertEqual(len(events), 10)
            self.assertEqual(len(items), 10)  # same numeric IDs stay domain-distinct
            rows[0]['test'][-1] = '3'
            lines('sequence_splits.jsonl', rows)
            with self.assertRaisesRegex(ValueError, 'split/order'):
                load_bundle(p, 'S', 'T')

    def test_mixed_old_embedding_tokenizer_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); p = paths(root, 'sports_to_clothing')
            p['processed'].mkdir(parents=True)
            p['tokenizer'].mkdir(parents=True)
            p['embedding'].mkdir(parents=True)
            (p['tokenizer']/'manifest.json').write_text(json.dumps({'embeddings_sha256': 'old', 'item_texts_sha256': 'old'}))
            (p['embedding']/'manifest.json').write_text(json.dumps({'input_sha256': 'old'}))
            (p['processed']/'item_texts.jsonl').write_text('new text')
            (p['embedding']/'text_embeddings.pt').write_bytes(b'new vectors')
            with self.assertRaisesRegex(RuntimeError, 'embedding|Tokenizer'):
                validate_asset_chain(root, 'sports_to_clothing')


if __name__ == '__main__': unittest.main()
