import copy
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class EightGPUQueueTests(unittest.TestCase):
    def card(self, n, busy=False):
        return dict(index=n, uuid=f'GPU-{n}', memory_free_mib=45000,
                    utilization=100 if busy else 0, pids=[999] if busy else [])

    def test_fills_idle_card_before_adopted_workers_finish_and_skips_done(self):
        from experiments.extend_amazon_recommendation import run_queue
        with tempfile.TemporaryDirectory() as tmp:
            active = {'old': True}
            done, calls = {('done', 42)}, []
            adopted = [dict(job=dict(pair='old', seed=42), gpu_uuid='GPU-1',
                            process=dict(pid=10, start_ticks=1))]
            def complete(job):
                return (job['pair'], job['seed']) in done
            def worker(job, card):
                self.assertTrue(active['old'])
                calls.append((job['pair'], card['index']))
                done.add((job['pair'], job['seed']))
                active['old'] = False
                done.add(('old', 42))
            run_queue([dict(pair=p, seed=42) for p in ('done', 'old', 'new')], worker, Path(tmp),
                adopted=adopted, alive=lambda p: active['old'], inventory=lambda: [self.card(1, True), self.card(7)],
                resources=lambda n: None, completed=complete, poll=.001)
            self.assertEqual(calls, [('new', 7)])

    def test_affinity_and_busy_cards_are_respected(self):
        from experiments.extend_amazon_recommendation import choose_assignment
        pending = [dict(pair='a', seed=42, gpu_uuid='GPU-6'), dict(pair='b', seed=42)]
        choice = choose_assignment(pending, [self.card(6, True), self.card(7)], set())
        self.assertEqual((choice[0]['pair'], choice[1]['index']), ('b', 7))
        self.assertIsNone(choose_assignment(pending[:1], [self.card(6, True), self.card(7)], set()))
        self.assertIsNone(choose_assignment(pending, [self.card(8)], set()))

    def test_eight_worker_cap_counts_adopted_and_reserves_host_memory(self):
        from experiments.extend_amazon_recommendation import run_queue
        with tempfile.TemporaryDirectory() as tmp:
            state = {'live': True}
            completed = set()
            reservations = []
            adopted = [dict(job=dict(pair=f'old{i}', seed=42), gpu_uuid=f'GPU-{i}',
                            process=dict(pid=10+i, start_ticks=1)) for i in range(7)]
            def work(job, card):
                self.assertEqual(card['index'], 7)
                completed.update(f'old{i}' for i in range(7))
                completed.add('new')
                state['live'] = False
            run_queue([r['job'] for r in adopted]+[dict(pair='new', seed=42)], work, Path(tmp),
                adopted=adopted, alive=lambda p: state['live'], inventory=lambda: [self.card(i) for i in range(8)],
                resources=reservations.append, completed=lambda j: j['pair'] in completed, poll=.001)
            self.assertEqual(reservations, [7*24*2**30])

    def test_failed_adopted_worker_prevents_any_new_admission(self):
        from experiments.extend_amazon_recommendation import run_queue
        with tempfile.TemporaryDirectory() as tmp:
            worker = Mock()
            with self.assertRaisesRegex(RuntimeError, 'without verified completion'):
                run_queue([dict(pair=p, seed=42) for p in ('old', 'new')], worker, Path(tmp),
                    adopted=[dict(job=dict(pair='old', seed=42), gpu_uuid='GPU-1', process=dict(pid=10))],
                    alive=lambda p: False, inventory=lambda: [self.card(7)], resources=Mock(),
                    completed=lambda j: False, poll=.001)
            worker.assert_not_called()


class HandoffTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'real Linux controller/child adoption')
    def test_linux_pidfd_retirement_keeps_real_worker_child_alive(self):
        import os
        import subprocess
        from experiments.extend_amazon_recommendation import handoff_controller
        from experiments.extend_gencdr_run import PidfdBackend, process_snapshot
        backend = PidfdBackend()
        script = (
            'import subprocess,sys,time; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
            'print(p.pid,flush=True); time.sleep(30)'
        )
        controller = subprocess.Popen([sys.executable, '-u', '-c', script], stdout=subprocess.PIPE, text=True)
        descriptor = None
        try:
            worker_pid = int(controller.stdout.readline())
            worker = process_snapshot(worker_pid)
            descriptor = backend.open(worker_pid)
            records = [dict(process=worker, job=dict(pair='fixture', seed=42), gpu_uuid='fixture')]
            result = handoff_controller(process_snapshot(controller.pid), backend=backend,
                snapshot=process_snapshot, discover=lambda: records, persist=Mock())
            self.assertEqual(result, records)
            self.assertEqual(controller.wait(timeout=5), -signal.SIGTERM)
            self.assertIsNotNone(process_snapshot(worker_pid))
        finally:
            if descriptor is not None:
                try:
                    backend.send(descriptor, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                os.close(descriptor)
            if controller.poll() is None:
                controller.terminate()
            controller.wait(timeout=5)
            controller.stdout.close()

    def test_pre_exec_child_is_settled_before_handoff_can_retire_controller(self):
        from experiments.extend_amazon_recommendation import settle_workers
        state = {'exec_complete': False}
        child = dict(process=dict(pid=51), job=dict(pair='p', seed=42), gpu_uuid='GPU-7')
        result = settle_workers(lambda: [child] if state['exec_complete'] else [],
            children=lambda: {51}, exited=lambda pid: False,
            sleep=lambda delay: state.update(exec_complete=True))
        self.assertEqual(result, [child])
        with self.assertRaisesRegex(RuntimeError, 'did not settle'):
            settle_workers(lambda: [], lambda: {51}, lambda pid: False, sleep=Mock(), attempts=2)

    def test_controller_relative_paths_use_its_own_cwd_and_wrong_cwd_is_rejected(self):
        from experiments.extend_amazon_recommendation import Extension
        import os
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            ext = Extension.__new__(Extension)
            ext.root, ext.batch, ext.alignment = base/'target', 'batch', base/'alignment.json'
            ext.args = SimpleNamespace(original_tool=base/'tool')
            ext.old = SimpleNamespace(DEFAULT_BATCH_ID='default')
            process = dict(uid=1, cwd=str(base/'tool'), argv=['python', 'runner.py', '--root', '../target',
                '--batch-id', 'batch', '--alignment', '../alignment.json'])
            ext.proc = SimpleNamespace(script_matches=Mock(return_value=True), owned_processes=lambda: [process])
            with patch.object(os, 'getuid', return_value=1, create=True):
                self.assertTrue(ext.matches(process))
                self.assertEqual(ext.controllers(), [process])
                process['cwd'] = str(base/'other')
                with self.assertRaisesRegex(RuntimeError, 'working directory'):
                    ext.controllers()
                process['argv'][process['argv'].index('../target')] = 'target'
                self.assertFalse(ext.matches(process))

    def test_equals_worker_argument_cannot_be_mistaken_for_controller(self):
        from experiments.extend_amazon_recommendation import cli_option
        self.assertEqual(cli_option(['python', 'run.py', '--worker=validation'], '--worker'), 'validation')
        self.assertIsNone(cli_option(['python', 'run.py'], '--worker'))
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            cli_option(['--worker=test', '--worker', 'validation'], '--worker')

    def test_only_controller_is_signalled_and_children_are_adopted(self):
        from experiments.extend_amazon_recommendation import handoff_controller, SIGSTOP, SIGCONT
        process = dict(pid=50, start_ticks=5, uid=1, cwd='/tool', argv=['python', 'run.py'], state='S')
        current = {'value': copy.deepcopy(process)}
        signals, receipts = [], []
        def send(fd, sig):
            signals.append((fd, sig))
            if sig == SIGSTOP:
                current['value']['state'] = 'T'
            elif sig == SIGCONT:
                current['value'] = None
        backend = Mock(open=Mock(return_value=9), send=Mock(side_effect=send))
        children = [dict(process=dict(pid=51, start_ticks=6), job=dict(pair='pair', seed=42), gpu_uuid='GPU-7')]
        result = handoff_controller(process, backend=backend, snapshot=lambda pid: current['value'],
            discover=lambda: children, persist=lambda value: receipts.append(value), close=Mock(), sleep=Mock())
        self.assertEqual(result, children)
        self.assertEqual([s for _, s in signals], [SIGSTOP, signal.SIGTERM, SIGCONT])
        self.assertTrue(all(fd == 9 for fd, _ in signals))
        self.assertTrue(receipts)

    def test_reused_controller_pid_is_never_signalled(self):
        from experiments.extend_amazon_recommendation import handoff_controller
        process = dict(pid=50, start_ticks=5, uid=1, cwd='/tool', argv=['python', 'run.py'], state='S')
        backend = Mock(open=Mock(return_value=9))
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            handoff_controller(process, backend=backend, snapshot=lambda pid: {**process, 'start_ticks': 6},
                discover=Mock(), persist=Mock(), close=Mock(), sleep=Mock())
        backend.send.assert_not_called()

    def test_capture_failure_resumes_controller_and_never_terminates_it(self):
        from experiments.extend_amazon_recommendation import handoff_controller, SIGSTOP, SIGCONT
        process = dict(pid=50, start_ticks=5, uid=1, cwd='/tool', argv=['python', 'run.py'], state='T')
        backend = Mock(open=Mock(return_value=9))
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
            handoff_controller(process, backend=backend, snapshot=lambda pid: process,
                discover=Mock(side_effect=RuntimeError('ambiguous worker')), persist=Mock(), close=Mock(), sleep=Mock())
        self.assertEqual([call.args[1] for call in backend.send.call_args_list], [SIGSTOP, SIGCONT])

    def test_interrupted_handoff_reuses_historical_receipt_when_controller_is_stopped(self):
        from experiments.extend_amazon_recommendation import Extension
        with tempfile.TemporaryDirectory() as tmp:
            ext = Extension.__new__(Extension)
            ext.extension = Path(tmp)
            process = dict(pid=50, start_ticks=5, uid=1, cwd='/tool', argv=['python', 'run.py'], state='S')
            import json
            path = Path(tmp)/'handoff.json'
            path.write_text(json.dumps(dict(controller=process, adopted=[])))
            before = path.read_bytes()
            ext.old = SimpleNamespace(read=lambda p: json.loads(p.read_text()), bind=Mock())
            ext.persist_handoff(dict(controller={**process, 'state': 'T'}, adopted=[{'new': 'snapshot'}]))
            self.assertEqual(path.read_bytes(), before)
            ext.old.bind.assert_not_called()


class BaselineCapacityTests(unittest.TestCase):
    def test_gencdr_waiting_scheduler_accepts_eight_workers(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, 'experiments/run_gencdr_baseline.py',
                '--root', tmp, '--action', 'status', '--max-workers', '8'], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


class CanonicalPipelineTests(unittest.TestCase):
    def test_eight_gpu_queue_preserves_original_validation_freeze_tests_and_restart(self):
        from experiments import run_amazon_recommendation as original
        from experiments.extend_amazon_recommendation import run_queue
        from test_amazon_recommendation_runner import AmazonRecommendationRunnerTests
        def expanded(group, worker, status, **kwargs):
            kwargs['max_workers'] = 8
            return run_queue(group, worker, status, adopted=[], alive=lambda p: False,
                             writer=original.write_json_atomic, **kwargs)
        fixture = AmazonRecommendationRunnerTests(
            'test_controller_runs_twelve_validations_before_test_and_restart_reuses_everything')
        with patch.object(original, 'run_queue', expanded):
            fixture.test_controller_runs_twelve_validations_before_test_and_restart_reuses_everything()


if __name__ == '__main__':
    unittest.main()
