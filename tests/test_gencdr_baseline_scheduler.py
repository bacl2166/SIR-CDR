import json
from pathlib import Path
import tempfile
import unittest

from cdr_framework.datasets.gencdr_protocol import sha
from cdr_framework.baselines.gencdr.protocol import jobs, key, sir_completion


class GenCDRSchedulerTests(unittest.TestCase):
    def test_sir_completion_accepts_all_twelve_only_and_rejects_changed_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/'artifacts/amazon_recommendation/source'
            out.mkdir(parents=True)
            def write(path, payload):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding='utf-8')
            write(out/'declaration.json', {})
            rows, bindings = [], []
            for j in jobs():
                marker = out/'jobs'/j['pair']/f"seed{j['seed']}"/'training/manifest.json'
                write(marker, dict(complete=True))
                bindings.append(dict(**j, files={str(marker.relative_to(root)): sha(marker)},
                                     test_identity=dict(**j, split='test')))
                rows.append(dict(**j, metrics={'HR@5': .1, 'NDCG@5': .05, 'HR@10': .2, 'NDCG@10': .1}))
            write(out/'validation_freeze.json', dict(schema='amazon-12-job-validation-freeze-v1', test_released=True, jobs=bindings,
                declaration_sha256=sha(out/'declaration.json')))
            for j, row in zip(bindings, rows):
                report = out/'jobs'/j['pair']/f"seed{j['seed']}"/'training/test.json'
                write(report, dict(state='complete', split='test', metrics=row['metrics'], recovery_identity=j['test_identity']))
                write(out/'test_completions'/f'{key(j)}.json', dict(**{k:j[k] for k in ('pair', 'seed')},
                    test_identity=j['test_identity'], report_sha256=sha(report), freeze_sha256=sha(out/'validation_freeze.json')))
            write(out/'results.json', dict(per_seed=rows, freeze_sha256=sha(out/'validation_freeze.json')))
            self.assertEqual(len(sir_completion(root, 'source')), 27)
            original = (out/'validation_freeze.json').read_bytes()
            invalid = json.loads(original)
            invalid['jobs'][0]['files'] = {}
            write(out/'validation_freeze.json', invalid)
            altered_result = dict(per_seed=rows, freeze_sha256=sha(out/'validation_freeze.json'))
            write(out/'results.json', altered_result)
            for receipt in (out/'test_completions').glob('*.json'):
                value = json.loads(receipt.read_text())
                value['freeze_sha256'] = sha(out/'validation_freeze.json')
                write(receipt, value)
            with self.assertRaisesRegex(RuntimeError, 'validation provenance'):
                sir_completion(root, 'source')
            (out/'validation_freeze.json').write_bytes(original)
            write(out/'results.json', dict(per_seed=rows, freeze_sha256=sha(out/'validation_freeze.json')))
            for receipt in (out/'test_completions').glob('*.json'):
                value = json.loads(receipt.read_text())
                value['freeze_sha256'] = sha(out/'validation_freeze.json')
                write(receipt, value)
            report.write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
                sir_completion(root, 'source')

    def test_gpu_selection_requires_empty_card_and_can_return_no_match(self):
        from experiments.run_gencdr_baseline import eligible_cards
        cards = [dict(uuid='a', pids=[1], utilization=0, memory_free_mib=45000, index=0),
                 dict(uuid='b', pids=[], utilization=0, memory_free_mib=24000, index=1),
                 dict(uuid='c', pids=[], utilization=0, memory_free_mib=45000, index=2)]
        self.assertEqual([c['uuid'] for c in eligible_cards(cards)], ['c'])
        self.assertEqual(eligible_cards(cards, reserved={'c'}), [])

    def test_scheduler_waits_without_loading_backbone_or_querying_gpu_until_sir_done(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from experiments import run_gencdr_baseline as runner
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(root=Path(tmp), batch_id='baseline', source_batch='sir',
                config=None, max_workers=2, poll_seconds=60)
            with patch.object(runner, 'source_identity', return_value={'fixture': True}), \
                 patch.object(runner, 'local_model_seal') as model, \
                 patch.object(runner, 'query_gpu_inventory') as inventory, \
                 patch.object(runner.time, 'sleep', side_effect=RuntimeError('stop waiting')):
                with self.assertRaisesRegex(RuntimeError, 'stop waiting'):
                    runner.run(args)
                model.assert_not_called()
                inventory.assert_not_called()
                self.assertTrue((runner.folder(args.root, 'baseline')/'request.json').exists())


if __name__ == '__main__':
    unittest.main()
