from pathlib import Path
import tempfile
import unittest


class SixRunnerTests(unittest.TestCase):
    def test_smoke_gate_rejects_insufficient_memory_margin(self):
        from experiments.run_gencdr_six import training_memory_requirement
        report={'peak_cuda_GiB':10.0,'optimizer_steps':0}
        self.assertEqual(training_memory_requirement(report), 14848)
        with self.assertRaises(ValueError): training_memory_requirement({'peak_cuda_GiB':None})

    def test_preparation_commands_do_not_train_or_test_recommender(self):
        from experiments.run_gencdr_six import asset_commands
        commands=asset_commands(Path('/repo'),'sports_to_clothing')
        self.assertEqual([c[c.index('--action')+1] for c in commands],['embedding','tokenize'])
        self.assertTrue(all('--asset-worker' in c for c in commands))

    def test_validation_jobs_are_eighteen_and_seed42_first(self):
        from experiments.run_gencdr_six import validation_jobs, worker_commands
        jobs = validation_jobs()
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({(j['pair'], j['seed']) for j in jobs}), 18)
        self.assertEqual([j['seed'] for j in jobs[:6]], [42]*6)
        commands = worker_commands(Path('/repo'), 'paper-v1', jobs[0], 'run-validation')
        self.assertEqual([c[c.index('--action')+1] for c in commands], ['train', 'validate'])
        self.assertTrue(all('--batch-id' in c for c in commands))

    def test_validation_failure_preserves_siblings_and_stops_later_seeds(self):
        from experiments.run_gencdr_six import run_validation_queue
        seen = []
        def queue(jobs, **kwargs):
            seen.extend(jobs)
            return [{'pair': j['pair'], 'seed': j['seed'], 'returncode': int(i == 0)} for i,j in enumerate(jobs)]
        with self.assertRaises(RuntimeError):
            run_validation_queue(queue)
        self.assertEqual(len(seen), 6)
        self.assertEqual({j['seed'] for j in seen}, {42})

    def test_queue_uses_each_approved_card_once_and_records_errors(self):
        from experiments.run_gencdr_six import schedule_jobs
        from cdr_framework.experiment_control import read_json
        import threading
        active = set(); guard = threading.Lock()
        def worker(job, gpu):
            with guard:
                self.assertNotIn(gpu, active); active.add(gpu)
            try:
                if job['pair'] == 'bad': raise RuntimeError('fixture failure')
                return {'complete': True}
            finally:
                with guard: active.remove(gpu)
        jobs = [{'pair': p, 'seed': 42} for p in ('a','bad','c')]
        inventory = lambda: [{'index': g, 'uuid': f'GPU-{g}', 'memory_free_mib': 40000,
                              'utilization': 0, 'pids': []} for g in (1,3)]
        with tempfile.TemporaryDirectory() as folder:
            results = schedule_jobs(jobs, [1,3], 2, Path(folder), worker,
                                    inventory=inventory, resource_check=lambda: None)
            self.assertEqual(sum(r['returncode'] != 0 for r in results), 1)
            self.assertEqual(len(list(Path(folder).glob('*.json'))), 3)
            self.assertEqual({r['gpu'] for r in results}, {1,3})

    def test_busy_card_is_not_launched(self):
        from experiments.run_gencdr_six import schedule_jobs
        launched = []
        inventory = lambda: [{'index': 0,'uuid': 'GPU-a','memory_free_mib': 40000,
                              'utilization': 50,'pids':[99]}]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                schedule_jobs([{'pair':'a','seed':42}], [0], 1, Path(folder),
                              lambda *args: launched.append(args), inventory=inventory,
                              resource_check=lambda: None)
        self.assertEqual(launched, [])


if __name__ == '__main__': unittest.main()
