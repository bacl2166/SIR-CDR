import json
from pathlib import Path
import tempfile
import unittest

import torch

from cdr_framework.datasets.gencdr_protocol import Event, build_shared, sha


def fixture(root):
    from cdr_framework.baselines.gencdr.protocol import PAIRS
    events = [Event('u', d, str(i), i*10+offset, f'{d}{i}')
              for d, offset in (('A', 0), ('B', 5)) for i in range(1, 7)]
    items = {(e.domain, e.item): e.item for e in events}
    for pair, source, target in ((PAIRS[0], 'A', 'B'), (PAIRS[1], 'B', 'A')):
        processed = root / 'data/processed/gencdr-shared-loo-v1' / pair
        build_shared(events, items, source, target, processed, provenance={'fixture': True})
        mapping = json.loads((processed/'mappings.json').read_text())
        vectors = torch.randn(len(mapping['id_to_item']), 6)
        vectors[0].zero_()
        output = root/'artifacts/embeddings/gencdr-shared-loo-v1'/pair
        output.mkdir(parents=True)
        torch.save(vectors, output/'text_embeddings.pt')
        (output/'manifest.json').write_text(json.dumps(dict(matrix_shape=list(vectors.shape),
            item_count=len(vectors)-1, input_sha256=sha(processed/'item_texts.jsonl'),
            batch_size=2, completed_batches=(len(vectors)-1+1)//2)))


class GenCDRTrainingTests(unittest.TestCase):
    def test_tokenizer_epoch_resume_matches_uninterrupted_in_both_phases(self):
        from cdr_framework.baselines.gencdr import training
        from cdr_framework.baselines.gencdr.assets import ExistingAssets
        from unittest.mock import patch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            assets = ExistingAssets(root, 'food_to_kitchen')
            cfg = training.BaselineConfig(hidden=(12, 8), latent=4, levels=2, codebook_size=4,
                tokenizer_rank=2, tokenizer_epochs=2, tokenizer_domain_epochs=2, tokenizer_batch_size=4)
            identity = {'assets': assets.seal()}
            complete = root/'token_complete'
            complete.mkdir()
            training.seed_all(42)
            training.fit_tokenizer(assets, complete, cfg, identity, torch.device('cpu'))
            expected = torch.load(complete/'tokenizer.pt', weights_only=True)
            real_save = training._save_atomic
            for interrupt_after in (1, 3):
                output = root/f'token_resumed_{interrupt_after}'
                output.mkdir()
                calls = []
                def interrupt(path, value):
                    real_save(path, value)
                    calls.append(1)
                    if len(calls) == interrupt_after:
                        raise RuntimeError('simulated tokenizer interruption')
                training.seed_all(42)
                with patch.object(training, '_save_atomic', interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'simulated tokenizer'):
                        training.fit_tokenizer(assets, output, cfg, identity, torch.device('cpu'))
                training.seed_all(42)
                training.fit_tokenizer(assets, output, cfg, identity, torch.device('cpu'))
                actual = torch.load(output/'tokenizer.pt', weights_only=True)
                self.assertTrue(torch.equal(expected['codes'], actual['codes']))
                for name, value in expected['model'].items():
                    self.assertTrue(torch.equal(value, actual['model'][name]), (interrupt_after, name))

    def test_tiny_real_training_resume_freeze_and_test_preserve_all_inputs(self):
        from cdr_framework.baselines.gencdr.training import BaselineConfig, train_job, evaluate_job
        from cdr_framework.baselines.gencdr.assets import ExistingAssets
        from transformers import Qwen3Config, Qwen3Model
        torch.set_num_threads(1)
        def factory():
            return Qwen3Model(Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            files = {str(p): sha(p) for p in root.rglob('*') if p.is_file()}
            cfg = BaselineConfig(hidden=(12, 8), latent=4, levels=2, codebook_size=4,
                tokenizer_rank=2, tokenizer_epochs=1, tokenizer_domain_epochs=1,
                universal_epochs=1, domain_epochs=2, rank=2, alpha=4, experts=2,
                batch_size=2, tokenizer_batch_size=4, beam_size=20, patience=2)
            pair = 'food_to_kitchen'
            assets = ExistingAssets(root, pair)
            output = root/'baseline'
            identity = {'fixture': True, 'assets': assets.seal()}
            manifest = train_job(root, pair, 42, output, cfg, identity, torch.device('cpu'), factory)
            self.assertTrue(manifest['complete'])
            self.assertFalse((output/'test.json').exists())
            before = sha(output/'best.pt')
            again = train_job(root, pair, 42, output, cfg, identity, torch.device('cpu'), factory)
            self.assertEqual(again, manifest)
            self.assertEqual(sha(output/'best.pt'), before)
            with self.assertRaisesRegex(RuntimeError, 'freeze'):
                evaluate_job(root, pair, 42, output, cfg, identity, torch.device('cpu'), None, factory)
            freeze = {'checkpoint_sha256': before, 'identity': identity}
            report = evaluate_job(root, pair, 42, output, cfg, identity, torch.device('cpu'), freeze, factory)
            self.assertEqual(report['split'], 'test')
            self.assertTrue(all(0 <= report['metrics'][m] <= 1 for m in ('HR@5', 'NDCG@5', 'HR@10', 'NDCG@10')))
            self.assertEqual(files, {p: sha(p) for p in files})
            changed = assets.processed/'statistics.json'
            original = changed.read_bytes()
            changed.write_bytes(original+b' ')
            with self.assertRaisesRegex(RuntimeError, 'Bound input changed'):
                train_job(root, pair, 42, output, cfg, identity, torch.device('cpu'), factory)
            changed.write_bytes(original)

    def test_interrupted_epoch_resumes_to_same_selected_weights(self):
        from cdr_framework.baselines.gencdr.training import BaselineConfig, train_job
        from cdr_framework.baselines.gencdr.assets import ExistingAssets
        from transformers import Qwen3Config, Qwen3Model
        from unittest.mock import patch
        from cdr_framework.baselines.gencdr import training
        torch.set_num_threads(1)
        def factory():
            return Qwen3Model(Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            cfg = BaselineConfig(hidden=(12, 8), latent=4, levels=2, codebook_size=4,
                tokenizer_rank=2, tokenizer_epochs=1, tokenizer_domain_epochs=1,
                universal_epochs=1, domain_epochs=2, rank=2, alpha=4, experts=2,
                batch_size=2, tokenizer_batch_size=4, beam_size=20, patience=2)
            assets = ExistingAssets(root, 'food_to_kitchen')
            identity = {'fixture': True, 'assets': assets.seal()}
            first, second = root/'uninterrupted', root/'resumed'
            train_job(root, 'food_to_kitchen', 42, first, cfg, identity, torch.device('cpu'), factory)
            real_evaluate = training.evaluate
            calls = []
            def interrupt(*args):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError('simulated interruption')
                return real_evaluate(*args)
            with patch.object(training, 'evaluate', interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    train_job(root, 'food_to_kitchen', 42, second, cfg, identity, torch.device('cpu'), factory)
            train_job(root, 'food_to_kitchen', 42, second, cfg, identity, torch.device('cpu'), factory)
            a = torch.load(first/'best.pt', weights_only=True)
            b = torch.load(second/'best.pt', weights_only=True)
            self.assertEqual(a['validation'], b['validation'])
            for name, value in a['model'].items():
                self.assertTrue(torch.equal(value, b['model'][name]), name)


if __name__ == '__main__':
    unittest.main()
