import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'experiments/expand_sports_hpo.py'


def load():
    spec = importlib.util.spec_from_file_location('sports_idle_pool_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu(number):
    return f'GPU-{number:08x}-1111-2222-3333-444444444444'


class IdlePoolTests(unittest.TestCase):
    def test_unknown_study_worker_blocks_new_id_or_checkout_restart(self):
        tool = load()
        expansion = tool.Expansion.__new__(tool.Expansion)
        expansion.root, expansion.original_root = Path('/train'), Path('/original')
        expansion.folder, expansion.path = Path('/study'), Path('/idle-gpus-v2')
        expansion.previous_root, expansion.previous_id = Path('/previous'), 'gpu7-v1'
        for script, identifier in ((SOURCE, 'another-expansion'),
                                   (Path('/other/experiments/expand_sports_hpo.py'), 'idle-gpus-v2')):
            process = dict(pid=123456789, uid=42, start_ticks=1, cwd=str(script.parent.parent),
                argv=['python', str(script), '--action', 'worker', '--root', '/train',
                      '--study-id', 'study', '--expansion-id', identifier])
            with patch.object(tool.base.support, 'owned_processes', return_value=[process]):
                with self.assertRaisesRegex(RuntimeError, 'Unknown'):
                    expansion.actors()

    def test_status_can_list_successor_but_run_rejects_second_controller(self):
        tool = load()
        expansion = tool.Expansion.__new__(tool.Expansion)
        expansion.root, expansion.original_root = Path('/train'), Path('/original')
        expansion.folder, expansion.path = Path('/study'), Path('/idle-gpus-v2')
        expansion.previous_root, expansion.previous_id = Path('/previous'), 'gpu7-v1'
        process = dict(pid=123456789, uid=42, start_ticks=1, cwd=str(tool.ROOT),
            argv=['python', str(SOURCE), '--action', 'run', '--root', '/train', '--study-id', 'study'])
        with patch.object(tool.base.support, 'owned_processes', return_value=[process]):
            with self.assertRaises(RuntimeError):
                expansion.actors()
            self.assertEqual(expansion.actors(allow_successor=True), ([process], []))

    def test_worker_observation_ignores_transient_process_state(self):
        tool = load()
        old = tool.base.load_file('idle_observation_fixture', SOURCE.parent / 'tune_sports_clothing.py')
        with tempfile.TemporaryDirectory() as directory:
            expansion = tool.Expansion.__new__(tool.Expansion)
            expansion.old, expansion.path = old, Path(directory)
            process = dict(pid=123, uid=42, start_ticks=1, cwd='/train',
                           argv=['python', '--job', 'sports_to_clothing/c03/seed42'], state='R', ppid=77)
            expansion.remember_worker(process)
            expansion.remember_worker(dict(process, state='S', ppid=1))
            records = list((expansion.path / 'observed_workers').glob('*.json'))
            self.assertEqual(len(records), 1)
            self.assertNotIn('state', old.read_record(records[0]))

    def test_expansion_keeps_original_policy_and_validates_budgets(self):
        tool = load()
        old = dict(gpus=[gpu(5), gpu(6), gpu(7)], max_workers=2,
                   host_ram_gib=64., job_ram_gib=24.)
        expanded = tool.resources_for(old, [gpu(i) for i in range(5)], 6, 192.)
        self.assertEqual(expanded['gpus'], old['gpus'] + [gpu(i) for i in range(5)])
        self.assertEqual(expanded['max_workers'], 6)
        self.assertEqual(expanded['job_ram_gib'], 24.)
        self.assertEqual(old['max_workers'], 2)
        for additions, cap, ram in (([gpu(5)], 6, 192.), ([gpu(0)] * 2, 2, 64.),
                                    (['0'], 2, 64.), ([gpu(0)], 5, 96.),
                                    ([gpu(0)], 2, float('nan')),
                                    ([gpu(i) for i in range(5)], 7, 192.)):
            with self.assertRaises(ValueError):
                tool.resources_for(old, additions, cap, ram)

    def test_only_expected_predecessor_can_be_signalled(self):
        tool = load()
        root, previous = Path('/train'), Path('/control')
        process = dict(pid=123, uid=42, start_ticks=1, cwd=str(previous),
            argv=['python', str(previous / 'experiments/extend_sports_hpo.py'),
                  '--action', 'run', '--root', str(root), '--add-gpu', gpu(7)])
        tool.validate_predecessor(process, root, previous, 'sports-clothing-ndcg-v1', 'gpu7-v1', 42)
        for change in ({'uid': 99}, {'argv': process['argv'] + ['--extension-id', 'other']},
                       {'argv': [v if v != 'run' else 'worker' for v in process['argv']]}):
            with self.assertRaises(RuntimeError):
                tool.validate_predecessor(dict(process, **change), root, previous,
                    'sports-clothing-ndcg-v1', 'gpu7-v1', 42)

    def test_handoff_finishes_worker_before_retiring_only_controller(self):
        tool = load()
        old = tool.base.load_file('idle_old_fixture', SOURCE.parent / 'tune_sports_clothing.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = dict(pid=123, uid=42, start_ticks=1, state='S')
            child = dict(pid=124, uid=42, start_ticks=2, state='R', cwd=str(root),
                         argv=['python', '--job', 'clothing_to_sports/c03/seed42', '--phase', 'train'])
            expansion = tool.Expansion.__new__(tool.Expansion)
            expansion.old, expansion.path = old, root
            events, state = [], {'controller': True, 'child': True}
            def actors():
                return ([controller] if state['controller'] else [], [child] if state['child'] else [])
            def send(process, signum):
                self.assertEqual(process['pid'], 123)
                if signum == 19:
                    events.append('stop'); controller['state'] = 'T'
                elif signum == 15:
                    self.assertFalse(state['child']); events.append('term')
                elif signum == 18:
                    events.append('cont'); state['controller'] = False
            def drain():
                events.append('drain'); state['child'] = False
            with patch.object(expansion, 'actors', side_effect=actors), \
                    patch.object(expansion, 'validate_target'), \
                    patch.object(expansion, 'drain', side_effect=drain), \
                    patch.object(tool.base, 'send_bound', side_effect=send), \
                    patch.object(tool.base.support, 'process_snapshot',
                        side_effect=lambda pid: controller if state['controller'] else None), \
                    patch.object(tool.signal, 'SIGSTOP', 19, create=True), \
                    patch.object(tool.signal, 'SIGTERM', 15, create=True), \
                    patch.object(tool.signal, 'SIGCONT', 18, create=True):
                expansion.handoff()
            self.assertEqual(events, ['stop', 'drain', 'term', 'cont'])
            self.assertTrue((root / 'retiring.json').is_file())
            self.assertTrue((root / 'retired.json').is_file())

    def test_schedule_skips_done_and_dispatches_idle_cards_to_cap(self):
        tool = load()
        old = tool.base.load_file('idle_schedule_fixture', SOURCE.parent / 'tune_sports_clothing.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expansion = tool.Expansion.__new__(tool.Expansion)
            expansion.old, expansion.folder, expansion.path = old, root / 'study', root / 'extension'
            expansion.root = root
            expansion.previous_root, expansion.previous_id = root / 'previous', 'gpu7-v1'
            jobs, done, dispatched = old.policy.screen_jobs()[:7], set(), []
            done.add(old.job_key(jobs[0]))
            pinned = expansion.folder / 'jobs' / old.job_key(jobs[1]) / 'affinity.json'
            old.seal(pinned, {'gpu_uuid': gpu(6)})
            before = pinned.read_bytes()
            resources = dict(gpus=[gpu(i) for i in range(8)], max_workers=6,
                             host_ram_gib=192., job_ram_gib=24.)
            cards = [dict(uuid=gpu(i), free_mib=45000, utilization=0.) for i in range(8)]
            class Process:
                pid = 123
                returncode = 0
                def poll(self): return 0
                def wait(self): return 0
            def popen(command, **kwargs):
                key = command[command.index('--job') + 1]
                dispatched.append((key, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                done.add(key)
                return Process()
            def complete(job, phase, modules):
                return dict(job, files={}) if old.job_key(job) in done else None
            def sleep(_):
                # A busy pinned job must remain pending until its original GPU clears.
                busy.clear()
            busy = {gpu(6), gpu(7)}
            with patch.object(expansion, 'completion', side_effect=complete), \
                    patch.object(old, 'gpu_snapshot', side_effect=lambda: (cards, set(busy))), \
                    patch.object(old, 'available_ram', return_value=400.), \
                    patch.object(tool.subprocess, 'Popen', side_effect=popen), \
                    patch.object(tool.time, 'sleep', side_effect=sleep):
                expansion.schedule(jobs, 'train', {}, resources)
            self.assertEqual(len(dispatched), 6)
            self.assertNotIn(old.job_key(jobs[0]), [key for key, _ in dispatched])
            self.assertEqual(dict(dispatched)[old.job_key(jobs[1])], gpu(6))
            self.assertEqual(pinned.read_bytes(), before)
            self.assertEqual(len(list((expansion.path / 'completed_jobs').glob('*.json'))), 7)

    def test_full_study_uses_original_budget_and_selection_and_can_be_repeated_read_only(self):
        tool = load()
        old = tool.base.load_file('idle_full_fixture', SOURCE.parent / 'tune_sports_clothing.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expansion = tool.Expansion.__new__(tool.Expansion)
            expansion.old, expansion.root = old, root
            expansion.folder, expansion.path = root / 'study', root / 'extension'
            old.seal(expansion.folder / 'study.json', {'unchanged': True})
            old.seal(expansion.folder / 'resource_policy.json', {'unchanged': True})
            old.seal(expansion.path / 'resources.json', {'new': True})
            old.seal(expansion.path / 'takeover.json', {'new': True})
            protected = {p: p.read_bytes() for p in expansion.folder.glob('*.json')}
            trained, scored, executed = {}, {}, []
            def metrics(job):
                return {name: .1 + int(job['candidate'][1:]) * .01 for name in old.policy.METRICS}
            for job in old.policy.screen_jobs()[:7]:
                trained[old.job_key(job)] = dict(job, metrics=metrics(job), files={})
            def completion(job, phase, modules):
                return (trained if phase == 'train' else scored).get(old.job_key(job))
            def schedule(jobs, phase, modules, resources):
                for job in jobs:
                    key = old.job_key(job)
                    if phase == 'train' and key not in trained:
                        executed.append(key)
                        trained[key] = dict(job, metrics=metrics(job), files={})
                    elif phase == 'validate' and key not in scored:
                        scored[key] = dict(job, results=[dict(setting=s, metrics=metrics(job))
                            for s in old.policy.fusion_settings()])
                        old.seal(expansion.folder / 'jobs' / key / 'selection.json', scored[key])
            with patch.object(old, 'original', return_value={}), \
                    patch.object(expansion, 'completion', side_effect=completion), \
                    patch.object(expansion, 'schedule', side_effect=schedule), \
                    patch.object(expansion, 'verify', return_value=({'prior_observations': {}}, {})):
                expansion.continue_study({})
                freeze = (expansion.folder / 'validation_freeze.json').read_bytes()
                expansion.continue_study({})
                self.assertEqual((expansion.folder / 'validation_freeze.json').read_bytes(), freeze)
            self.assertEqual(len(executed), 17)
            self.assertEqual(len(set(executed)), 17)
            final = old.read(expansion.folder / 'validation_freeze.json')
            self.assertEqual(len(final['all_training']), 24)
            self.assertEqual(len(final['selection_receipts']), 12)
            self.assertFalse(final['test_action'])
            for path, before in protected.items():
                self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
