import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
import threading
import time

from experiments.extend_gencdr_run import (
    immutable_receipt, eligible_jobs, validate_controller, identify_job,
    check_training_completion, verify_extension_freeze, terminate_controller, Coordinator,
)


PAIRS = ('a', 'b', 'c', 'd', 'e', 'f')
SEEDS = (42, 43, 44)


class ExtensionTests(unittest.TestCase):
    def test_completed_seed_is_not_repeated_and_next_seed_can_fill_slot(self):
        jobs = eligible_jobs(PAIRS, SEEDS, {('a', 42)}, {('b', 42), ('c', 42)})
        self.assertEqual(jobs[:4], [('d', 42), ('e', 42), ('f', 42), ('a', 43)])
        self.assertNotIn(('a', 42), jobs)
        self.assertEqual(len(jobs), 15)

    def test_remaining_independent_seed_can_fill_otherwise_idle_slot(self):
        self.assertEqual(eligible_jobs(('a',), SEEDS, {('a', 42)}, {('a', 43)}), [('a', 44)])

    def test_all_eighteen_done_leaves_no_training_jobs(self):
        done = {(p, s) for p in PAIRS for s in SEEDS}
        self.assertEqual(eligible_jobs(PAIRS, SEEDS, done, set()), [])

    def test_completed_later_seed_cannot_hide_missing_earlier_seed(self):
        self.assertEqual(eligible_jobs(('a',), SEEDS, {('a', 43)}, set()), [('a', 42), ('a', 44)])

    def test_resource_receipt_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            immutable_receipt(path, {'gpus': [0, 1, 4, 5, 6, 7]})
            immutable_receipt(path, {'gpus': [0, 1, 4, 5, 6, 7]})
            with self.assertRaises(RuntimeError):
                immutable_receipt(path, {'gpus': [0, 1, 2, 3, 4, 5]})
            self.assertEqual(json.loads(path.read_text())['gpus'], [0, 1, 4, 5, 6, 7])

    def controller(self, **changes):
        value = {'pid': 123, 'uid': 1000, 'start_ticks': 777, 'cwd': '/repo',
                 'argv': ['python', '-u', 'experiments/run_gencdr_six.py',
                          '--action', 'run-validation', '--batch-id', 'paper-v2']}
        value.update(changes)
        return value

    def test_controller_requires_exact_owner_root_batch_and_action(self):
        validate_controller(self.controller(), Path('/repo'), 'paper-v2', 1000)
        for value in (self.controller(uid=1001), self.controller(cwd='/other'),
                      self.controller(argv=['python', '-c', 'experiments/run_gencdr_six.py']),
                      self.controller(argv=['python', 'experiments/run_gencdr_six.py',
                                            '--action', 'freeze', '--batch-id', 'paper-v2'])):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                validate_controller(value, Path('/repo'), 'paper-v2', 1000)

    def test_wrapper_and_orphan_inner_process_map_to_same_job(self):
        root = Path('/repo').resolve()
        runtime = root / 'configs/runtime/a_seed42.yaml'
        configs = {str(runtime): ('a', 42)}
        wrapper = {'cwd': str(root), 'argv': ['python', str(root / 'experiments/run_gencdr_shared.py'),
                   '--action', 'train', '--pairs', 'a', '--seed', '42', '--batch-id', 'paper-v2']}
        inner = {'cwd': str(root), 'argv': ['python', str(root / 'experiments/run_llm_cdr.py'),
                 '--action', 'train', '--config', str(runtime)]}
        self.assertEqual(identify_job(wrapper, root, 'paper-v2', configs), ('a', 42))
        self.assertEqual(identify_job(inner, root, 'paper-v2', configs), ('a', 42))
        inner['argv'][-1] = str(root / 'configs/runtime/other.yaml')
        self.assertIsNone(identify_job(inner, root, 'paper-v2', configs))

    def test_unrelated_python_process_is_not_adopted(self):
        process = {'cwd': '/repo', 'argv': ['python', '/elsewhere/run_llm_cdr.py', '--config', '/repo/a.yaml']}
        self.assertIsNone(identify_job(process, Path('/repo'), 'v2', {'/repo/a.yaml': ('a', 42)}))

    def test_training_skip_requires_complete_and_checkpoint_consistency(self):
        identity = {'config': {'seed': 42}}
        state = {'identity': identity, 'epoch': 2, 'validation': {'HR@10': .1}}
        manifest = {'identity': identity, 'complete': True, 'epochs_completed': 3,
                    'best_validation': {'HR@10': .1}}
        check_training_completion(manifest, state, identity)
        for change in ({'complete': False}, {'epochs_completed': 1}, {'best_validation': {'HR@10': .2}},
                       {'identity': {'wrong': True}}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                check_training_completion(manifest | change, state, identity)

    def test_extension_freeze_requires_matching_both_hashes(self):
        receipt = {'resource_sha256': 'r', 'validation_freeze_sha256': 'f'}
        verify_extension_freeze(receipt, 'r', 'f')
        for resource, freeze in (('other', 'f'), ('r', 'other')):
            with self.assertRaises(RuntimeError):
                verify_extension_freeze(receipt, resource, freeze)

    def test_pidfd_handoff_signals_only_verified_controller(self):
        process = self.controller()
        with patch('experiments.extend_gencdr_run.os.pidfd_open', return_value=99, create=True) as opened, \
                patch('experiments.extend_gencdr_run.os.getuid', return_value=1000, create=True), \
                patch('experiments.extend_gencdr_run.os.close') as closed, \
                patch('experiments.extend_gencdr_run.signal.pidfd_send_signal', create=True) as sent, \
                patch('experiments.extend_gencdr_run.process_snapshot', side_effect=[process, None]):
            terminate_controller(process, Path('/repo'), 'paper-v2')
            opened.assert_called_once_with(123)
            import signal
            sent.assert_called_once_with(99, signal.SIGTERM)
            closed.assert_called_once_with(99)

    def test_pid_reuse_and_wrong_owner_are_never_signalled(self):
        process = self.controller()
        for changed in (process | {'start_ticks': 888}, process | {'uid': 1001}):
            with self.subTest(changed=changed), \
                    patch('experiments.extend_gencdr_run.os.pidfd_open', return_value=99, create=True), \
                    patch('experiments.extend_gencdr_run.os.getuid', return_value=1000, create=True), \
                    patch('experiments.extend_gencdr_run.os.close'), \
                    patch('experiments.extend_gencdr_run.signal.pidfd_send_signal', create=True) as sent, \
                    patch('experiments.extend_gencdr_run.process_snapshot', return_value=changed):
                with self.assertRaises(RuntimeError):
                    terminate_controller(process, Path('/repo'), 'paper-v2')
                sent.assert_not_called()

    def test_two_independent_training_processes_for_one_job_are_not_adopted(self):
        coordinator = Coordinator.__new__(Coordinator)
        coordinator.maximum = 6
        coordinator.affinity = lambda job: 'GPU-a'
        coordinator.inventory = lambda: [{'uuid': 'GPU-a', 'pids': [101, 102]}]
        rows = [{'pid': 101, 'ppid': 1, 'gpu_uuid': 'GPU-a'},
                {'pid': 102, 'ppid': 1, 'gpu_uuid': 'GPU-a'}]
        with self.assertRaises(RuntimeError):
            coordinator.validate_adoption({('a', 42): rows})

    def test_one_wrapper_and_its_inner_process_can_be_adopted(self):
        coordinator = Coordinator.__new__(Coordinator)
        coordinator.maximum = 6
        coordinator.affinity = lambda job: 'GPU-a'
        coordinator.inventory = lambda: [{'uuid': 'GPU-a', 'pids': [102]}]
        rows = [{'pid': 101, 'ppid': 1, 'gpu_uuid': 'GPU-a'},
                {'pid': 102, 'ppid': 101, 'gpu_uuid': 'GPU-a'}]
        coordinator.validate_adoption({('a', 42): rows})

    def test_completed_job_is_not_executed_again(self):
        coordinator = Coordinator.__new__(Coordinator)
        coordinator.wait_existing = Mock()
        coordinator.root = Path('/repo')
        coordinator.batch = 'batch'
        coordinator.core = SimpleNamespace(SCHEMA='schema')
        coordinator.completed = Mock(return_value={'verified': True})
        coordinator.stage = Mock()
        coordinator.work(('a', 42), 'GPU-a', adopted=True)
        coordinator.wait_existing.assert_called_once_with(('a', 42))
        coordinator.stage.assert_not_called()

    def test_complete_training_with_missing_validation_runs_only_validation(self):
        coordinator = Coordinator.__new__(Coordinator)
        identity = {'config': {'seed': 42}}
        state = {'identity': identity, 'epoch': 1, 'validation': {'HR@10': .1}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'manifest.json').write_text(json.dumps({'identity': identity, 'complete': True,
                'epochs_completed': 2, 'best_validation': state['validation']}))
            coordinator.root, coordinator.batch = output, 'batch'
            coordinator.core = SimpleNamespace(SCHEMA='schema')
            coordinator.configs = {('a', 42): SimpleNamespace(output_dir=output)}
            coordinator.completed = Mock(side_effect=[None, {'verified': True}])
            coordinator.tuning = SimpleNamespace(validate_checkpoint=Mock(return_value=(state, identity)))
            coordinator.stage = Mock()
            coordinator.work(('a', 42), 'GPU-a')
            self.assertEqual([c.args[1] for c in coordinator.stage.call_args_list], ['validate'])

    def test_failed_freeze_guard_cannot_start_test(self):
        coordinator = Coordinator.__new__(Coordinator)
        coordinator.root, coordinator.batch = Path('/repo'), 'batch'
        coordinator.core = SimpleNamespace(SCHEMA='schema')
        coordinator.guard_freeze = Mock(side_effect=RuntimeError('changed freeze'))
        coordinator.stage = Mock()
        with self.assertRaisesRegex(RuntimeError, 'changed freeze'):
            coordinator.work(('a', 42), 'GPU-a', testing=True)
        coordinator.stage.assert_not_called()

    def test_handoff_recovers_controller_exit_before_marker_without_signalling_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = Coordinator.__new__(Coordinator)
            coordinator.control = coordinator.root = Path(directory)
            coordinator.resource = coordinator.control / 'resource_extension.json'
            coordinator.resource.write_text('{"resource": "fixed"}')
            coordinator.configs = {}
            coordinator.batch = 'batch'
            coordinator.controller = Mock(side_effect=AssertionError('must not seek a new controller'))
            from experiments.extend_gencdr_run import sha
            (coordinator.control / 'extension_handoff_intent.json').write_text(json.dumps({
                'controller': {'pid': 123, 'start_ticks': 777},
                'resource_sha256': sha(coordinator.resource)}))
            with patch('experiments.extend_gencdr_run.process_snapshot', return_value=None), \
                    patch('experiments.extend_gencdr_run.terminate_controller') as terminate:
                coordinator.handoff()
                terminate.assert_not_called()

    def test_existing_handoff_can_resume_after_canonical_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = Coordinator.__new__(Coordinator)
            coordinator.control = coordinator.root = Path(directory)
            coordinator.resource = coordinator.control / 'resource_extension.json'
            coordinator.resource.write_text('{"resource": "fixed"}')
            from experiments.extend_gencdr_run import sha
            (coordinator.control / 'extension_handoff_complete.json').write_text(json.dumps({
                'resource_sha256': sha(coordinator.resource)}))
            (coordinator.control / 'validation_freeze.json').write_text('{"frozen": true}')
            coordinator.handoff()

    def test_initial_takeover_is_closed_if_test_was_already_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = Coordinator.__new__(Coordinator)
            coordinator.control = Path(directory)
            coordinator.resource = coordinator.control / 'resource_extension.json'
            coordinator.configs = {}
            (coordinator.control / 'validation_freeze.json').write_text('{"frozen": true}')
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                coordinator.handoff()

    def test_six_slot_queue_reuses_done_seed_prefers_directions_and_fills_with_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = Coordinator.__new__(Coordinator)
            coordinator.root = coordinator.control = Path(directory)
            coordinator.batch = 'batch'
            coordinator.pairs, coordinator.seeds = PAIRS, SEEDS
            coordinator.maximum, coordinator.poll, coordinator.gpus = 6, .001, tuple(range(6))
            coordinator.configs = {(p, s): None for p in PAIRS for s in SEEDS}
            initial = {('a', 42)}
            coordinator.completed = lambda j: j in initial
            coordinator.live = lambda: {}
            coordinator.pre_admission = lambda: None
            coordinator.write_status = Mock()
            coordinator.core = SimpleNamespace(
                admit_gpu_whitelist=lambda ids, cards: {'allowed_gpus': list(ids)},
                assert_gpu_launchable=lambda *args: None,
                host_resources=lambda: {'memory': {'MemAvailable': 256 * 2**30}},
                utc_now=lambda: 'now')
            coordinator.runner = SimpleNamespace(training_memory_requirement=lambda report: 4096,
                                                 admit_host_budget=lambda **kwargs: None)
            cards = [{'index': i, 'uuid': f'GPU-{i}', 'pids': [], 'memory_free_mib': 40000} for i in range(6)]
            coordinator.inventory = lambda: cards
            for pair in PAIRS:
                path = coordinator.control / 'smoke' / f'{pair}.json'
                path.parent.mkdir(exist_ok=True)
                path.write_text('{"report": {}}')
            guard, in_flight, calls, finished = threading.Lock(), set(), [], set(initial)
            def work(job, uuid, adopted=False, testing=False):
                with guard:
                    self.assertNotIn(job, in_flight)
                    self.assertNotIn(job, finished)
                    in_flight.add(job)
                    self.assertLessEqual(len(in_flight), 6)
                    calls.append(job)
                time.sleep(.002)
                with guard:
                    in_flight.remove(job)
                    finished.add(job)
            coordinator.work = work
            coordinator.queue()
            self.assertEqual(len(calls), 17)
            self.assertNotIn(('a', 42), calls)
            self.assertEqual({p for p, s in calls[:6]}, set(PAIRS))
            self.assertEqual(finished, set(coordinator.configs))


if __name__ == '__main__':
    unittest.main()
