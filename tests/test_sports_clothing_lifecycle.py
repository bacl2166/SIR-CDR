"""Exercise complete orchestration and selection without CUDA or real datasets."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'experiments/tune_sports_clothing.py'
spec = importlib.util.spec_from_file_location('sports_lifecycle_controller', SOURCE)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def metrics(base):
    return dict(zip(controller.policy.METRICS,
                    (base*.6, base*.4, base, base*.7, base*1.2, base*.8)))


class StudyLifecycleTests(unittest.TestCase):
    def exercise(self, directory, missing_selection=False):
        root = Path(directory)
        folder = root / 'artifacts/hparam_search/fixture-study'
        folder.mkdir(parents=True)
        controller.seal(folder / 'study.json', {'fixture': True})
        original = root / 'original-v2-control.json'
        original.write_text('{"unchanged": true}')
        original_hash = controller.sha(original)
        args = SimpleNamespace(
            gpus='GPU-11111111-1111-1111-1111-111111111111,GPU-22222222-2222-2222-2222-222222222222',
            max_workers=2, host_ram_gib=64., job_ram_gib=24.)
        trained, scored, executions, phases = {}, {}, [], []

        def schedule(root, folder, jobs, phase, modules, resources):
            phases.append((phase, list(jobs)))
            for job in jobs:
                key = controller.job_key(job)
                if phase == 'train' and key not in trained:
                    executions.append(key)
                    path = folder / 'jobs' / key / 'training/manifest.json'
                    controller.seal(path, job)
                    trained[key] = dict(job, metrics=metrics(.1 + int(job['candidate'][1:])*.01),
                                        files={str(path.relative_to(folder)): controller.sha(path)})
                elif phase == 'validate' and key not in scored:
                    # Screening favors c07, but multi-seed snapshot ranking favors c06.
                    base = .15 + (.01 if job['candidate'] == 'c06' else 0) + (job['seed']-42)*.001
                    results = []
                    for setting in controller.policy.fusion_settings():
                        bonus = .02 if setting == {'pool': 1000, 'weight': .875, 'normalization': 'none'} else 0
                        results.append({'setting': setting, 'metrics': metrics(base + bonus)})
                    path = folder / 'jobs' / key / 'selection.json'
                    controller.seal(path, dict(job, results=results))
                    scored[key] = dict(job, results=results)

        def get_scored(folder, job, config, modules):
            if missing_selection and job['seed'] == 44 and job['pair'] == controller.policy.PAIRS[-1]:
                return None
            return scored.get(controller.job_key(job))

        with patch.object(controller, 'orphan_workers'), \
                patch.object(controller, 'verify_study', return_value={'prior_observations': {'fixture': True}}), \
                patch.object(controller, 'original', return_value={}), \
                patch.object(controller, 'config_for', side_effect=lambda root, folder, job, modules: job), \
                patch.object(controller, 'trained', side_effect=lambda folder, job, config, modules: trained.get(controller.job_key(job))), \
                patch.object(controller, 'scored', side_effect=get_scored), \
                patch.object(controller, 'schedule', side_effect=schedule):
            if missing_selection:
                with self.assertRaises((ValueError, RuntimeError)):
                    controller.run(root, folder, args)
                self.assertFalse((folder / 'validation_freeze.json').exists())
                return
            controller.run(root, folder, args)
            frozen_hash = controller.sha(folder / 'validation_freeze.json')
            result = controller.read(folder / 'validation_freeze.json')
            self.assertEqual(len(executions), 24)
            self.assertEqual(len(set(executions)), 24)
            self.assertEqual(len(scored), 12)
            self.assertEqual([len(jobs) for _, jobs in phases], [16, 8, 12])
            self.assertEqual(len(result['all_training']), 24)
            self.assertEqual(len(result['selection_receipts']), 12)
            self.assertFalse(result['test_action'])
            for direction in result['validation']['directions']:
                self.assertEqual(direction['candidate'], 'c06')
                self.assertEqual(direction['setting'], {'pool': 1000, 'weight': .875, 'normalization': 'none'})
                self.assertEqual(direction['seeds'], [42, 43, 44])
                self.assertGreater(direction['metrics']['N@10']['std'], 0)
            controller.run(root, folder, args)
            self.assertEqual(len(executions), 24)
            self.assertEqual(controller.sha(folder / 'validation_freeze.json'), frozen_hash)
            self.assertEqual(controller.sha(original), original_hash)

    def test_full_staged_study_freezes_all_provenance_and_resumes_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            self.exercise(directory)

    def test_incomplete_finalist_validation_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            self.exercise(directory, missing_selection=True)


if __name__ == '__main__':
    unittest.main()
