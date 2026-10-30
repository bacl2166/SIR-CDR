import importlib.util
from pathlib import Path
import tempfile
import os
import sys
import subprocess
import signal
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / 'experiments/extend_sports_hpo.py'


def load():
    spec = importlib.util.spec_from_file_location('sports_extension_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GPUResourceExtensionTests(unittest.TestCase):
    def test_scheduler_skips_done_job_and_dispatches_to6_and7_preserving_affinity(self):
        tool = load()
        old = tool.load_file('old_scheduler_fixture', SOURCE.parent / 'tune_sports_clothing.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, extension = root / 'study', root / 'extension'
            jobs = old.policy.screen_jobs()[:3]
            gpus = [f'GPU-{n*8}-{n*4}-{n*4}-{n*4}-{n*12}' for n in ('5', '6', '7')]
            resources = dict(gpus=gpus, max_workers=2, host_ram_gib=64., job_ram_gib=24.)
            old.seal(folder / 'jobs' / old.job_key(jobs[1]) / 'affinity.json', {'gpu_uuid': gpus[1]})
            rows = {old.job_key(jobs[0]): dict(jobs[0], metrics={'N@10': .1}, files={})}
            dispatched = []
            class Child:
                returncode = None
                pid = 123
                def __init__(self, command):
                    self.key = tool.support.option(command, '--job')
                    gpu = tool.support.option(command, '--gpu')
                    dispatched.append((self.key, gpu))
                def poll(self):
                    self.returncode = 0
                    rows[self.key] = dict(old.job_from_key(self.key), metrics={'N@10': .1}, files={})
                    return 0
                def wait(self):
                    return self.poll()
            def complete(old_arg, root_arg, folder_arg, job, phase, modules):
                return rows.get(old.job_key(job))
            cards = [dict(uuid=g, free_mib=24000, utilization=0) for g in gpus]
            with patch.object(old, 'gpu_snapshot', return_value=(cards, {gpus[0]})), \
                    patch.object(old, 'available_ram', return_value=128), \
                    patch.object(tool, 'completion', side_effect=complete), \
                    patch.object(tool.subprocess, 'Popen', side_effect=lambda command, **kw: Child(command)), \
                    patch.object(tool.time, 'sleep'):
                tool.schedule(old, root, folder, extension, jobs, 'train', {}, resources)
            self.assertEqual(dispatched, [(old.job_key(jobs[1]), gpus[1]), (old.job_key(jobs[2]), gpus[2])])
            self.assertEqual(old.read(folder / 'jobs' / old.job_key(jobs[1]) / 'affinity.json'), {'gpu_uuid': gpus[1]})
            self.assertEqual(len(list((extension / 'completed_jobs').glob('*.json'))), 3)

    def test_full_study_continuation_preserves_old_files_and_never_retrains_done_jobs(self):
        tool = load()
        old = tool.load_file('old_study_fixture', SOURCE.parent / 'tune_sports_clothing.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, extension = root / 'study', root / 'extension'
            old.seal(folder / 'study.json', {'original': True})
            old.seal_record(folder / 'resource_policy.json', {'gpus': ['old']})
            before = {p.name: p.read_bytes() for p in folder.glob('*.json')}
            old.seal_record(extension / 'resources.json', {'gpus': ['old', 'new']})
            old.seal_record(extension / 'takeover.json', {'transferred': True})
            train, scores, executed = {}, {}, []
            def metrics(job):
                return {k: .1 + int(job['candidate'][1:]) * .01 for k in old.policy.METRICS}
            for job in old.policy.screen_jobs()[:2]:
                train[old.job_key(job)] = dict(job, metrics=metrics(job), files={})
            def complete(old_arg, root_arg, folder_arg, job, phase, modules):
                return (train if phase == 'train' else scores).get(old.job_key(job))
            def schedule(old_arg, root_arg, folder_arg, extension_arg, jobs, phase, modules, resources):
                for job in jobs:
                    key = old.job_key(job)
                    if phase == 'train' and key not in train:
                        executed.append(key)
                        train[key] = dict(job, metrics=metrics(job), files={})
                    if phase == 'validate' and key not in scores:
                        scores[key] = dict(job, results=[{'setting': s, 'metrics': metrics(job)}
                                                        for s in old.policy.fusion_settings()])
                        old.seal(folder / 'jobs' / key / 'selection.json', scores[key])
            with patch.object(old, 'original', return_value={}), \
                    patch.object(tool, 'completion', side_effect=complete), \
                    patch.object(tool, 'schedule', side_effect=schedule), \
                    patch.object(tool, 'verify_extension', return_value=({'prior_observations': {'retained': True}}, {})):
                tool.continue_study(old, root, folder, extension, {})
                freeze = (folder / 'validation_freeze.json').read_bytes()
                tool.continue_study(old, root, folder, extension, {})
                self.assertEqual((folder / 'validation_freeze.json').read_bytes(), freeze)
            self.assertEqual(len(executed), 22)
            self.assertEqual(len(set(executed)), 22)
            final = old.read(folder / 'validation_freeze.json')
            self.assertEqual(len(final['all_training']), 24)
            self.assertEqual(len(final['selection_receipts']), 12)
            self.assertFalse(final['test_action'])
            for name, value in before.items():
                self.assertEqual((folder / name).read_bytes(), value)

    def test_cancellation_is_forbidden_once_retirement_has_started(self):
        tool = load()
        old = tool.load_file('old_retirement_fixture', SOURCE.parent / 'tune_sports_clothing.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'old-tool'
            controller = {'pid': 123, 'uid': 42, 'start_ticks': 100, 'state': 'T',
                'cwd': str(original), 'argv': ['python', 'experiments/tune_sports_clothing.py',
                    '--action', 'run', '--root', str(root), '--study-id', root.name]}
            old.seal_record(root / 'intent.json', {'controller': controller})
            old.seal_record(root / 'retiring.json', {'controller': controller})
            with patch.object(tool, 'send_bound') as send, \
                    patch.object(tool.support, 'process_snapshot', return_value=controller), \
                    patch.object(tool.os, 'getuid', return_value=42, create=True):
                with self.assertRaises(RuntimeError):
                    tool.resume_old(old, root, original, root, root)
                send.assert_not_called()
            self.assertFalse((root / 'resume_old.json').exists())

    def test_successor_drain_reports_incomplete_job_but_allows_checkpoint_resume(self):
        tool = load()
        old = tool.load_file('old_recovery_fixture', SOURCE.parent / 'tune_sports_clothing.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = old.policy.screen_jobs()[3]
            process = {'pid': 123, 'uid': 42, 'start_ticks': 100, 'cwd': str(root),
                       'argv': ['python', str(SOURCE), '--job', old.job_key(job), '--phase', 'train']}
            old.seal_record(root / 'quiesced.json', {'workers': [process]})
            with patch.object(tool, 'actors', return_value=([], [])), \
                    patch.object(tool, 'completion', return_value=None):
                with self.assertRaises(RuntimeError):
                    tool.drain_workers(old, root, root, root, root, {})
                tool.drain_workers(old, root, root, root, root, {}, require_success=False)
            self.assertEqual(len(list((root / 'incomplete_workers').glob('*.json'))), 1)

    def test_handoff_drains_worker_before_retirement_and_is_resumable(self):
        tool = load()
        old = tool.load_file('old_handoff_fixture', SOURCE.parent / 'tune_sports_clothing.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, original = root / 'study', root / 'old'
            extension = folder / 'resource_extensions/gpu7-v1'
            old.seal(folder / 'study.json', {'untouched': True})
            old.seal(folder / 'resource_policy.json', {'untouched': True})
            before = {p.name: p.read_bytes() for p in folder.glob('*.json')}
            controller = {'pid': 123, 'uid': 42, 'start_ticks': 100, 'state': 'S',
                'cwd': str(original), 'argv': ['python', 'experiments/tune_sports_clothing.py',
                    '--action', 'run', '--root', str(root), '--study-id', folder.name]}
            job = old.policy.screen_jobs()[2]
            child = {'pid': 124, 'uid': 42, 'start_ticks': 101, 'state': 'R',
                'cwd': str(root), 'argv': ['python', str(original / 'experiments/tune_sports_clothing.py'),
                    '--action', 'worker', '--root', str(root), '--study-id', folder.name,
                    '--job', old.job_key(job), '--phase', 'train']}
            state = {'controller': True, 'worker': True, 'complete': False}
            events = []
            def signal_process(process, signum):
                self.assertEqual(process['pid'], controller['pid'])
                if signum == 19:
                    events.append('controller_stopped'); controller['state'] = 'T'
                elif signum == 15:
                    self.assertTrue(state['complete'])
                    events.append('controller_retired')
                elif signum == 18:
                    state['controller'] = False
            def poll_sleep(seconds):
                if state['worker']:
                    events.append('worker_completed')
                    state.update(worker=False, complete=True)
            def actors(*args):
                return ([controller] if state['controller'] else [], [child] if state['worker'] else [])
            def completion(*args):
                return {'job': job, 'verified': True} if state['complete'] else None
            with patch.object(old, 'original', return_value={}), patch.object(tool, 'actors', side_effect=actors), \
                    patch.object(tool, 'send_bound', side_effect=signal_process), \
                    patch.object(tool, 'completion', side_effect=completion), \
                    patch.object(tool, 'verify_extension', return_value=({}, {})), \
                    patch.object(tool.os, 'getuid', return_value=42, create=True), \
                    patch.object(tool.signal, 'SIGSTOP', 19, create=True), \
                    patch.object(tool.signal, 'SIGCONT', 18, create=True), \
                    patch.object(tool.support, 'process_snapshot', side_effect=lambda pid: controller if state['controller'] else None), \
                    patch.object(tool.time, 'sleep', side_effect=poll_sleep):
                tool.drain_handoff(old, root, original, folder, extension)
                self.assertEqual(events, ['controller_stopped', 'worker_completed', 'controller_retired'])
                tool.drain_handoff(old, root, original, folder, extension)
                self.assertEqual(len(events), 3)
            self.assertTrue((extension / 'retired.json').is_file())
            self.assertFalse((extension / 'takeover.json').exists())
            self.assertEqual({p.name: p.read_bytes() for p in folder.glob('*.json')}, before)

    def test_frozen_file_changes_are_rejected_and_unmodified_files_are_preserved(self):
        tool = load()
        old = tool.load_file('old_fixture_control', SOURCE.parent / 'tune_sports_clothing.py')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ('study.json', 'study_hash.json', 'resource_policy.json'):
                (folder / name).write_text('{}')
            affinity = folder / 'jobs/sports_to_clothing/c01/seed42/affinity.json'
            old.seal(affinity, {'gpu_uuid': 'original'})
            hashes = tool.protected_hashes(old, folder)
            tool.verify_protected(old, folder, hashes)
            self.assertEqual(old.read(affinity), {'gpu_uuid': 'original'})
            (folder / 'resource_policy.json').write_text('{"changed":true}')
            with self.assertRaises(RuntimeError):
                tool.verify_protected(old, folder, hashes)

    @unittest.skipUnless(sys.platform == 'linux', 'real pidfd lifecycle requires Linux')
    def test_only_controller_is_stopped_while_child_completes(self):
        tool = load()
        code = ('import subprocess,sys,time; '
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(1)"]); '
                'print(p.pid,flush=True); time.sleep(60)')
        parent = subprocess.Popen([sys.executable, '-u', '-c', code], stdout=subprocess.PIPE, text=True)
        try:
            child_pid = int(parent.stdout.readline())
            snapshot = tool.support.process_snapshot(parent.pid)
            tool.send_bound(snapshot, signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while tool.support.process_snapshot(parent.pid)['state'] != 'T':
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.05)
            while tool.support.process_snapshot(child_pid) is not None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.05)
            self.assertEqual(tool.support.process_snapshot(parent.pid)['state'], 'T')
            tool.send_bound(snapshot, signal.SIGTERM)
            tool.send_bound(snapshot, signal.SIGCONT)
            parent.wait(timeout=5)
            self.assertNotEqual(parent.returncode, 0)
        finally:
            if parent.poll() is None:
                parent.kill(); parent.wait()
            parent.stdout.close()
    def test_pool_addition_preserves_caps_and_old_resource_file(self):
        tool = load()
        old = {'gpus': ['GPU-11111111-1111-1111-1111-111111111111'],
               'max_workers': 2, 'host_ram_gib': 64., 'job_ram_gib': 24.}
        extra = 'GPU-77777777-7777-7777-7777-777777777777'
        result = tool.extended_resources(old, extra)
        self.assertEqual(result, dict(old, gpus=old['gpus'] + [extra]))
        self.assertEqual(len(old['gpus']), 1)
        with self.assertRaises(ValueError):
            tool.extended_resources(old, old['gpus'][0])

    def test_controller_identity_cannot_match_a_worker_or_another_user(self):
        tool = load()
        root, original = Path('/data/train'), Path('/data/old-tool')
        process = {'pid': 123, 'uid': 42, 'start_ticks': 100, 'cwd': str(original),
                   'argv': ['python', 'experiments/tune_sports_clothing.py', '--action', 'run',
                            '--root', str(root), '--study-id', 'study']}
        tool.validate_controller(process, root, original, 'study', 42)
        with self.assertRaises(RuntimeError):
            tool.validate_controller(dict(process, uid=0), root, original, 'study', 42)
        with self.assertRaises(RuntimeError):
            changed = dict(process, argv=[v if v != 'run' else 'worker' for v in process['argv']])
            tool.validate_controller(changed, root, original, 'study', 42)
        relative = dict(process, argv=['python', 'experiments/tune_sports_clothing.py',
            '--action=run', '--root=../train', '--study-id=study'])
        tool.validate_controller(relative, root, original, 'study', 42)
        defaults = dict(process, argv=['python', 'experiments/tune_sports_clothing.py', '--action', 'run'])
        tool.validate_controller(defaults, Path('/Data/lihb/SIR-CDR'), original, 'sports-clothing-ndcg-v1', 42)

    def test_pid_reuse_prevents_even_controller_signal(self):
        tool = load()
        process = {'pid': 123, 'uid': 42, 'start_ticks': 100}
        backend = Mock()
        with patch.object(tool.support, 'process_snapshot', return_value=dict(process, start_ticks=101)), \
                patch.object(tool.os, 'close'):
            with self.assertRaises(RuntimeError):
                tool.send_bound(process, 19, backend)
        backend.send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
