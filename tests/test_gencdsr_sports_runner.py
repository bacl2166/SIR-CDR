import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class SportsRunnerTests(unittest.TestCase):
    def runner(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.run_gencdsr_sports'), 'staged sports driver is missing')
        from experiments import run_gencdsr_sports
        return run_gencdsr_sports

    def test_paths_and_recipe_are_separate_and_fixed(self):
        r = self.runner()
        for pair in r.PAIRS:
            cfg = r.model_config(Path('/new').resolve(), pair, 42, 'pilot')
            self.assertIn(r.SCHEMA, str(cfg.output_dir))
            self.assertEqual((cfg.batch_size, cfg.gradient_accumulation_steps, cfg.evaluation_batch_size), (8, 2, 8))
            self.assertEqual((cfg.selection_metric, cfg.reasoning_steps, cfg.lora_rank), ('NDCG@10', 2, 8))
            self.assertEqual((cfg.rerank_candidates, cfg.generation_score_weight, cfg.fusion_normalization), (500, .5, 'none'))
        for bad in ('../oops', '', 'a/b'):
            with self.assertRaises(ValueError): r.paths(Path('/new'), r.PAIRS[0], 42, bad)

    def test_immutable_receipts_reject_replacement(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'receipt.json'
            r.immutable(p, {'x': 1}); r.immutable(p, {'x': 1})
            with self.assertRaises(RuntimeError): r.immutable(p, {'x': 2})
            self.assertEqual(json.loads(p.read_text()), {'x': 1})

    def test_lock_excludes_second_owner_and_releases(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'job'
            with r.lock(p):
                with self.assertRaises(RuntimeError):
                    with r.lock(p): pass
            with r.lock(p): pass

    def test_gpu_admission_fails_closed(self):
        r = self.runner(); gpu = 'GPU-12345678-1234-1234-1234-123456789abc'
        card = {'uuid': gpu, 'free_mib': 12288, 'utilization': 0, 'pids': []}
        self.assertEqual(r.admit_gpu(gpu, [card], 24*2**30), card)
        for change in ({'pids':[99]}, {'utilization':1}, {'free_mib':12287}):
            with self.assertRaises(RuntimeError): r.admit_gpu(gpu, [{**card, **change}], 24*2**30)
        with self.assertRaises(RuntimeError): r.admit_gpu(gpu, [card], 24*2**30-1)
        with self.assertRaises(ValueError): r.admit_gpu('0', [card], 24*2**30)

    def test_sticky_affinity_cannot_move(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            r.affinity(Path(tmp), 'GPU-a')
            with self.assertRaises(RuntimeError): r.affinity(Path(tmp), 'GPU-b')

    def test_freeze_requires_six_actual_bindings_and_detects_tamper(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'audit_directions', return_value={'complete': True}):
            root = Path(tmp)
            with patch.object(r, 'binding', side_effect=FileNotFoundError):
                with self.assertRaises(FileNotFoundError): r.freeze(root, 'study')
            def bind(root, pair, seed, study): return {'pair':pair, 'seed':seed, 'actual':'abc'}
            with patch.object(r, 'binding', side_effect=bind):
                frozen = r.freeze(root, 'study')
                self.assertEqual(len(frozen['jobs']), 6)
                self.assertEqual(r.verify_freeze(root, 'study'), frozen)
            with patch.object(r, 'binding', return_value={'actual':'changed'}):
                with self.assertRaises(RuntimeError): r.verify_freeze(root, 'study')

    def test_test_phase_checks_freeze_before_gpu_or_model_import(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, 'gpu_session') as gpu:
            with self.assertRaises(FileNotFoundError):
                r.workflow(Path(tmp), Path('/old'), 'study', r.PAIRS[0], 42, 'test', None)
            gpu.assert_not_called()

    def test_plan_declares_reconstruction_without_auto_schedule(self):
        r = self.runner(); p = r.plan(Path('/new'), Path('/old'), 'study')
        self.assertEqual(p['seeds'], [42,43,44])
        self.assertFalse(p['automatic_launch'])
        self.assertIn('joint', p['filter'])
        self.assertFalse(p['paper_alignment_verified'])

    def checkpoint_fixture(self, root):
        import torch
        from types import SimpleNamespace
        cfg = SimpleNamespace(output_dir=root, max_epochs=30, selection_metric='NDCG@10', patience=5)
        checkpoint = {'identity':{'real':'identity'}, 'epoch':3, 'validation':{'NDCG@10':.5,'seconds':2}}
        manifest = {'identity':checkpoint['identity'], 'complete':True, 'epochs_completed':30,
                    'best_validation':checkpoint['validation']}
        torch.save(checkpoint,root/'best_model.pt')
        torch.save({'identity':checkpoint['identity'],'epoch':30,'stale':0},root/'last_checkpoint.pt')
        (root/'manifest.json').write_text(json.dumps(manifest))
        return cfg,checkpoint,manifest

    def test_completed_rejects_fake_early_completion_and_nonfinite_metric(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, checkpoint, manifest = self.checkpoint_fixture(root)
            with patch('cdr_framework.llm_training.signature',return_value=checkpoint['identity']):
                self.assertEqual(r.completed(cfg),checkpoint)
                manifest['epochs_completed'] = 3
                (root/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(RuntimeError): r.completed(cfg)

    def test_completed_rejects_checkpoint_identity_or_best_metric_swap(self):
        import torch
        r = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, checkpoint, manifest = self.checkpoint_fixture(root)
            with patch('cdr_framework.llm_training.signature',return_value=checkpoint['identity']):
                checkpoint['validation'] = {'NDCG@10':.9}
                torch.save(checkpoint,root/'best_model.pt')
                with self.assertRaises(RuntimeError): r.completed(cfg)

    def test_binding_rejects_replaced_validation_even_with_correct_checkpoint(self):
        from types import SimpleNamespace
        r = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pair = r.PAIRS[0]; p = r.paths(root,pair,42,'study')
            p['output'].mkdir(parents=True); p['job'].mkdir(parents=True); p['runtime'].parent.mkdir(parents=True)
            checkpoint = {'epoch':3,'validation':{'NDCG@10':.5,'seconds':2}}
            for f in (p['runtime'],p['job']/'runtime.json',p['output']/'manifest.json',p['output']/'best_model.pt'):
                f.write_text('{}')
            report = {'split':'validation','mode':'hybrid','epoch':3,'checkpoint_sha256':r.sha(p['output']/'best_model.pt'),
                      'metrics':{'NDCG@10':.5,'seconds':5},'evaluation_identity':{'bound':True}}
            (p['output']/'validation_hybrid.json').write_text(json.dumps(report))
            (p['job']/'validation.json').write_text(json.dumps({'report_sha256':'tampered'}))
            with patch.object(r,'verify_runtime',return_value=SimpleNamespace()), patch.object(r,'completed',return_value=checkpoint), \
                 patch.object(r,'source_identity',return_value={}), patch.object(r,'verify_ready',return_value={}), \
                 patch('cdr_framework.text_training.evaluation_identity',return_value={'bound':True}):
                with self.assertRaises(RuntimeError): r.binding(root,pair,42,'study')

    def test_driver_import_does_not_import_torch(self):
        import subprocess,sys
        r = self.runner()
        result = subprocess.run([sys.executable,'-c',
            "import sys; import experiments.run_gencdsr_sports; assert 'torch' not in sys.modules"],
            cwd=r.ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_failed_test_keeps_permanent_reservation_and_forbids_retry(self):
        from contextlib import contextmanager
        from types import SimpleNamespace
        r = self.runner()
        @contextmanager
        def session(*args,**kwargs): yield ('cuda:0',{})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pair = r.PAIRS[0]; p = r.paths(root,pair,42,'study')
            r.immutable(r.freeze_path(root,'study'),{'frozen':True})
            with patch.object(r,'gpu_session',side_effect=session), patch.object(r,'verify_runtime',return_value=SimpleNamespace()), \
                 patch.object(r,'completed'), patch.object(r,'verify_freeze',return_value={'frozen':True}), \
                 patch('cdr_framework.llm_training.evaluate_checkpoint',side_effect=RuntimeError('interrupted')) as evaluate:
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    r.workflow(root,Path('/old'),'study',pair,42,'test','GPU-aaa')
                self.assertTrue((p['job']/'test-reservation.json').exists())
                with self.assertRaisesRegex(RuntimeError,'no retry'):
                    r.workflow(root,Path('/old'),'study',pair,42,'test','GPU-aaa')
                self.assertEqual(evaluate.call_count,1)

    def test_freeze_fails_when_a_training_job_holds_its_lock(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with r.lock(r.paths(root,r.PAIRS[1],44,'study')['job']/'operation.lock'), \
                 patch.object(r,'binding') as bind:
                with self.assertRaises(RuntimeError): r.freeze(root,'study')
                bind.assert_not_called()

    def test_clean_detached_source_is_recorded(self):
        from types import SimpleNamespace
        r = self.runner()
        def git(command,**kwargs):
            values = {('branch','--show-current'):'', ('status','--porcelain'):'',
                      ('ls-files','*.py'):'', ('rev-parse','HEAD'):'a'*40}
            return SimpleNamespace(stdout=values[tuple(command[1:])])
        with patch.object(r.subprocess,'run',side_effect=git):
            self.assertEqual(r.source_identity(),{'commit':'a'*40,'branch':'','files':{}})


if __name__ == '__main__': unittest.main()
