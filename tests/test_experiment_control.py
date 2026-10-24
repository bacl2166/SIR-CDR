import json
from pathlib import Path
import tempfile
import unittest

from cdr_framework import experiment_control as ec


def validation_fixture(root, batch='paper-v1'):
    """Full file graph; only expensive existing checkpoint/asset validation is mocked by callers."""
    from experiments.run_gencdr_shared import model_config, runtime_path, paths
    from cdr_framework.llm_training import config_payload
    import yaml
    import torch
    declaration = {'commit': 'fixture-commit', 'batch_id': batch, 'pairs': list(ec.SIX_PAIRS),
                   'seeds': list(ec.SEEDS), 'gpus': [0], 'paper_alignment_verified': False}
    ec.write_json_atomic(ec.control_root(root,batch)/'experiment_manifest.json', declaration)
    for pair in ec.SIX_PAIRS:
        p = paths(root,pair)
        for file in (p['processed']/'shared_manifest.json',p['control']/'ready.json'):
            ec.write_json_atomic(file, {'fixture':True})
        for seed in ec.SEEDS:
            cfg=model_config(root,pair,seed,batch)
            runtime=runtime_path(root,pair,seed,batch); runtime.parent.mkdir(parents=True,exist_ok=True)
            runtime.write_text(yaml.safe_dump({'llm_recommendation':config_payload(cfg)}),encoding='utf-8')
            identity={'config':config_payload(cfg), 'fixture':True}
            cfg.output_dir.mkdir(parents=True,exist_ok=True)
            torch.save({'identity':identity,'epoch':1,'validation':{'HR@10':.2}},cfg.output_dir/'best_model.pt')
            ec.write_json_atomic(cfg.output_dir/'manifest.json',{'identity':identity,'complete':True,
                                 'epochs_completed':1,'best_validation':{'HR@10':.2}})
            tuning=root/'artifacts/tuning'/ec.SCHEMA/batch/pair/f'seed{seed}'
            tuning.mkdir(parents=True,exist_ok=True)
            for name in ('item_states.pt','validation_scores.pt'): (tuning/name).write_bytes(b'fixture')
            evaluation={'training_identity':identity,'checkpoint_sha256':ec.file_sha(cfg.output_dir/'best_model.pt'),
                'source_config':config_payload(cfg),'completed_manifest_sha256':ec.file_sha(cfg.output_dir/'manifest.json'),
                'runtime':{'uuid':'GPU-a','device':'cuda:0'}}
            search={'identity':evaluation,'complete':True,'split':'validation','snapshot_sha256':ec.file_sha(tuning/'item_states.pt'),
                    'cache_sha256':ec.file_sha(tuning/'validation_scores.pt'),'best':{'setting':{'weight':.5},'metrics':{'N@10':.1}}}
            ec.write_json_atomic(tuning/'search.json',search)
            binding={'identity':evaluation,'snapshot_sha256':search['snapshot_sha256'],
                     'cache_sha256':search['cache_sha256'],'search_sha256':ec.file_sha(tuning/'search.json')}
            ec.write_json_atomic(tuning/'verification.json',{'binding':binding,'raw_scores_equal':True,
                                 'rankings_equal':True,'verified_all_metrics':True})


def fixture_checkpoint(config, path):
    import torch
    state=torch.load(path,weights_only=True)
    return state,state['identity']


class ControlTests(unittest.TestCase):
    def test_completed_embedding_rejects_different_pinned_encoder(self):
        expected={'input_sha256':'texts','model_revision':'a'*40,'model_config_sha256':'cfg',
                  'dimension':768,'normalized':True,'max_sequence_length':512,'item_count':2}
        value={**expected,'batch_size':2,'completed_batches':1,'matrix_shape':[3,768]}
        ec.require_embedding_recipe(value,expected)
        value['model_revision']='b'*40
        with self.assertRaises(RuntimeError): ec.require_embedding_recipe(value,expected)

    def test_storage_check_includes_actual_cache_filesystem(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=2*2**30,total=100*2**30)):
                with self.assertRaises(RuntimeError): ec.assert_storage_budget(root,root/'cache',24*2**30,16*2**30)

    def test_fictitious_eighteen_job_freeze_is_rejected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'README.md').write_text('fixture')
            filemap={'README.md':ec.file_sha(root/'README.md')}
            value={'schema':'six-direction-freeze-v1','batch_id':'paper-v1','commit':'fixture-commit',
                   'test_released':True,'files':filemap,'jobs':[
                       {'pair':p,'seed':s,'files':filemap,'test_identity':{'fake':True}}
                       for p in ec.SIX_PAIRS for s in ec.SEEDS]}
            ec.write_json_atomic(ec.control_root(root,'paper-v1')/'validation_freeze.json',value)
            with patch.object(ec,'git_identity',return_value={'commit':'fixture-commit','status_porcelain':''}):
                with self.assertRaises((RuntimeError,FileNotFoundError)): ec.verify_freeze(root,'paper-v1')

    def test_summary_preserves_all_seeds_and_uses_sample_standard_deviation(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); validation_fixture(root)
            with patch('experiments.run_gencdr_shared.verify_ready'), \
                 patch('experiments.tune_llm_fusion.validate_checkpoint',side_effect=fixture_checkpoint), \
                 patch.object(ec,'git_identity',return_value={'commit':'fixture-commit','status_porcelain':''}):
                frozen=ec.freeze_batch(root,'paper-v1')
                for job in frozen['jobs']:
                    score={42:.1,43:.2,44:.3}[job['seed']]
                    ec.write_json_atomic(root/job['output']/'test.json',
                        {'state':'complete','split':'test','recovery_identity':job['test_identity'],
                         'metrics':{k:score for k in ('R@5','N@5','R@10','N@10')}})
                    ec.record_test_completion(root,'paper-v1',job)
                result=ec.summarize_batch(root,'paper-v1')
                self.assertEqual(len(result['per_seed']),18)
                self.assertAlmostEqual(result['directions'][0]['metrics']['R@5']['mean'],.2)
                self.assertAlmostEqual(result['directions'][0]['metrics']['R@5']['std'],.1)
                self.assertFalse(result['strictly_comparable'])
                self.assertEqual(result['directions'][0]['target_domain'],'Clothing')
                test=root/frozen['jobs'][0]['output']/'test.json'
                invalid=ec.read_json(test); invalid['recovery_identity']={}; ec.write_json_atomic(test,invalid)
                with self.assertRaises(RuntimeError): ec.summarize_batch(root,'paper-v1')

    def test_metric_edit_before_first_summary_is_rejected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); validation_fixture(root)
            with patch('experiments.run_gencdr_shared.verify_ready'), \
                 patch('experiments.tune_llm_fusion.validate_checkpoint',side_effect=fixture_checkpoint), \
                 patch.object(ec,'git_identity',return_value={'commit':'fixture-commit','status_porcelain':''}):
                job=ec.freeze_batch(root,'paper-v1')['jobs'][0]
                path=root/job['output']/'test.json'
                report={'state':'complete','split':'test','recovery_identity':job['test_identity'],
                        'metrics':{k:.1 for k in ('R@5','N@5','R@10','N@10')}}
                ec.write_json_atomic(path,report)
                ec.record_test_completion(root,'paper-v1',job)
                report['metrics']['R@5']=.8; ec.write_json_atomic(path,report)
                with self.assertRaises(RuntimeError): ec.verify_test_completion(root,'paper-v1',job)

    def test_embedding_reuse_requires_exact_text_and_recipe_identity(self):
        identity={'input_sha256':'texts','model_revision':'a'*40,'model_config_sha256':'cfg',
                  'dimension':768,'normalized':True,'max_sequence_length':512,'item_count':2}
        self.assertTrue(ec.can_reuse_embedding(identity,dict(identity)))
        for key in identity:
            changed=dict(identity); changed[key]='different'
            self.assertFalse(ec.can_reuse_embedding(identity,changed))
        self.assertFalse(ec.can_reuse_embedding({},{}))

    def test_freeze_requires_all_eighteen_and_detects_tamper(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); validation_fixture(root)
            with patch('experiments.run_gencdr_shared.verify_ready'), \
                 patch('experiments.tune_llm_fusion.validate_checkpoint',side_effect=fixture_checkpoint), \
                 patch.object(ec,'git_identity',return_value={'commit':'fixture-commit','status_porcelain':''}):
                frozen=ec.freeze_batch(root,'paper-v1')
                self.assertEqual(len(frozen['jobs']),18)
                self.assertEqual(ec.freeze_batch(root,'paper-v1'),frozen)
                self.assertEqual(ec.verify_freeze(root,'paper-v1'),frozen)
                from experiments.run_gencdr_shared import model_config
                (model_config(root,'sports_to_clothing',42,'paper-v1').output_dir/'best_model.pt').write_bytes(b'tamper')
                with self.assertRaises(RuntimeError): ec.verify_freeze(root,'paper-v1')

    def test_partial_or_unverified_batch_cannot_freeze(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); validation_fixture(root)
            report=root/'artifacts/tuning'/ec.SCHEMA/'paper-v1/douban_movie_book/seed44/verification.json'
            report.unlink()
            with patch('experiments.run_gencdr_shared.verify_ready'), \
                 patch('experiments.tune_llm_fusion.validate_checkpoint',side_effect=fixture_checkpoint), \
                 patch.object(ec,'git_identity',return_value={'commit':'fixture-commit','status_porcelain':''}):
                with self.assertRaises((FileNotFoundError,RuntimeError)): ec.freeze_batch(root,'paper-v1')
            self.assertFalse((ec.control_root(root,'paper-v1')/'validation_freeze.json').exists())

    def test_read_only_environment_reports_dirty_tree_without_exposing_environment(self):
        from unittest.mock import patch
        from subprocess import CompletedProcess
        def run(args, **kwargs):
            text = {'rev-parse': 'abc123', 'branch': 'main', 'status': ' M changed.py',
                    'remote': 'git@github.com:example/repo.git'}.get(args[1], 'fixture')
            return CompletedProcess(args, 0, text, '')
        cards = [{'index': 0, 'uuid': 'GPU-a', 'memory_free_mib': 40000, 'utilization': 0, 'pids': []}]
        with tempfile.TemporaryDirectory() as folder, patch.object(ec, 'query_gpu_inventory', return_value=cards):
            report = ec.capture_environment(Path(folder), Path(folder), [0], run=run)
        self.assertFalse(report['admission']['production_ready'])
        self.assertFalse(report['training_started'])
        self.assertIn('git', report)
        self.assertNotIn('environ', report)
        self.assertIn('driver',report)
        self.assertIn('bf16_supported',report['torch'])
        self.assertIn('assets',report)

    def test_gpu_whitelist_and_busy_card_fail_closed(self):
        self.assertEqual(ec.parse_gpu_ids(['0', '3']), (0, 3))
        for values in ([], ['0', '0'], ['-1'], ['x']):
            with self.assertRaises(ValueError):
                ec.parse_gpu_ids(values)
        cards = [{'index': 0, 'uuid': 'GPU-a', 'memory_free_mib': 40000,
                  'utilization': 0, 'pids': []}]
        self.assertEqual(ec.admit_gpu_whitelist([0], cards)['allowed_gpus'], [0])
        with self.assertRaises(ValueError):
            ec.admit_gpu_whitelist([1], cards)
        cards[0]['pids'] = [123]
        self.assertEqual(ec.admit_gpu_whitelist([0], cards)['allowed_gpus'], [])
        with self.assertRaises(RuntimeError):
            ec.assert_gpu_launchable({'requested_gpu_whitelist': [0], 'gpu_inventory': cards}, 0)

    def test_nvidia_csv_links_processes_by_uuid(self):
        def run(args, **kwargs):
            from subprocess import CompletedProcess
            value = ('0, GPU-a, NVIDIA L20, 49140, 100, 49040, 0, 35, 42\n'
                     '1, GPU-b, NVIDIA L20, 49140, 2000, 47140, 40, 45, 80\n')
            if '--query-compute-apps=gpu_uuid,pid,used_gpu_memory' in args:
                value = 'GPU-b, 123, 2000\n'
            return CompletedProcess(args, 0, value, '')
        cards = ec.query_gpu_inventory(run)
        self.assertEqual(cards[0]['pids'], [])
        self.assertEqual(cards[1]['pids'], [123])

    def test_atomic_json_compare_and_freeze_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root/'input'; file.write_text('original')
            dest = ec.control_root(root, 'paper-v1')/'validation_freeze.json'
            frozen = {'schema': 'six-direction-freeze-v1', 'batch_id': 'paper-v1',
                      'files': {'input': ec.file_sha(file)}, 'jobs': [], 'test_released': True}
            ec.write_json_atomic(dest, frozen)
            # A malformed/partial manifest must never authorize a test.
            with self.assertRaises(RuntimeError): ec.verify_freeze(root, 'paper-v1')
            file.write_text('changed')
            with self.assertRaises(RuntimeError): ec.verify_freeze(root, 'paper-v1')


if __name__ == '__main__': unittest.main()
