import importlib.util
import json
from pathlib import Path
from unittest.mock import patch, Mock

import unittest
import tempfile
from contextlib import contextmanager

@contextmanager
def raises(kind, match=None):
    try:
        yield
    except kind as error:
        if match and match not in str(error):
            raise AssertionError(str(error))
    else:
        raise AssertionError(f"Expected {kind}")


SOURCE = Path(__file__).parents[1] / 'experiments/tune_sports_clothing.py'
spec = importlib.util.spec_from_file_location('sports_controller', SOURCE)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def test_active_reservations_count_against_available_host_ram(self):
        cards = [dict(uuid='GPU-abc', free_mib=24000, utilization=0)]
        controller.admit('GPU-abc', cards, set(), 0, 64, 24, 30)
        with raises(RuntimeError):
            controller.admit('GPU-abc', cards, set(), 24, 64, 24, 30)

    def test_self_validating_receipt_is_one_atomic_publication_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resource_policy.json'
            payload = {'gpus': ['approved'], 'max_workers': 1}
            with patch.object(controller, 'seal', wraps=controller.seal) as publish:
                self.assertEqual(controller.seal_record(path, payload), payload)
                self.assertEqual(publish.call_count, 1)
            self.assertEqual(controller.read_record(path), payload)
            self.assertEqual(controller.seal_record(path, payload), payload)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            record = controller.read(path)
            record['payload']['max_workers'] = 2
            path.write_text(json.dumps(record), encoding='utf-8')
            with raises(RuntimeError, 'integrity'):
                controller.read_record(path)

    def test_plan_discloses_ancillary_search_without_expanding_selection(self):
        plan = controller.plan()
        self.assertEqual(plan['snapshot_exploration'], {
            'selection_records_per_model': 48,
            'ancillary_original_records_per_model': 17,
            'total_records_per_model': 65,
            'unique_settings_per_model': 52,
            'total_records_for_finalists': 780,
            'ancillary_settings_used_for_final_selection': False,
            'excludes_native_epoch_validation_and_prior_exploration': True,
        })
        self.assertEqual(len(plan['fusion_settings']), 48)

    def test_prior_diagnostics_and_baselines_are_bound_to_real_files(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / 'diagnostic.json').write_text('{}')
        prior = {'source_batch': 'v2', 'preserved_diagnostic_reports': ['diagnostic.json']}
        binding = {'files': {'baseline.pt': 'hash'}, 'output': 'baseline'}
        (root / 'baseline').mkdir()
        (root / 'baseline/history.json').write_text('[]')
        control = Mock()
        control.collect_validation_binding.return_value = binding
        result = controller.prior_bindings(root, prior, {'cdr_framework.experiment_control': control})
        assert result['files']['diagnostic.json'] == controller.sha(root / 'diagnostic.json')
        assert result['files']['baseline/history.json'] == controller.sha(root / 'baseline/history.json')
        assert control.collect_validation_binding.call_count == 2
        with raises(ValueError):
            controller.prior_bindings(root, dict(prior, preserved_diagnostic_reports=['../outside.json']), {})

    def test_no_test_guard_covers_unselected_jobs(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        path = folder / 'jobs/clothing_to_sports/c07/seed44/training/test.lock'
        path.parent.mkdir(parents=True)
        path.write_text('reserved')
        with raises(RuntimeError, 'test'):
            controller.reject_test_artifacts(folder)

    def test_gpu_utilization_and_invalid_measurements_fail_closed(self):
        for utilization in (6, float('nan'), None, -1, 101):
            with raises(RuntimeError):
                controller.admit('GPU-abc', [dict(uuid='GPU-abc', free_mib=24000, utilization=utilization)],
                                 set(), 0, 64, 24, 64)

    def test_affinity_busy_head_does_not_block_idle_gpu(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        jobs = controller.policy.screen_jobs()[:2]
        for job, gpu in zip(jobs, ('GPU-a', 'GPU-b')):
            controller.seal(folder / 'jobs' / controller.job_key(job) / 'affinity.json', {'gpu_uuid': gpu})
        resources = dict(gpus=['GPU-a', 'GPU-b'], host_ram_gib=64, job_ram_gib=24)
        cards = [dict(uuid=g, free_mib=24000, utilization=0) for g in resources['gpus']]
        with patch.object(controller, 'gpu_snapshot', return_value=(cards, {'GPU-a'})), \
             patch.object(controller, 'available_ram', return_value=64):
            assert controller.eligible_pending(folder, jobs, [], resources) == (jobs[1], 'GPU-b')

    def test_promotions_are_recomputed_from_all_screen_artifacts(self):
        metrics = {k: .2 for k in controller.policy.METRICS} | {'examples': 100}
        rows = [dict(j, metrics=metrics, files={}) for j in controller.policy.screen_jobs()]
        expected = {'promotions': controller.policy.promote(rows), 'screen_rows': rows}
        with patch.object(controller, 'trained', side_effect=rows), \
             patch.object(controller, 'config_for'), patch.object(controller, 'read_record', return_value=expected):
            assert controller.promotions_for(None, Path('/study'), {}) == expected['promotions']
        tampered = dict(expected, promotions={p: ['c06', 'c07'] for p in controller.policy.PAIRS})
        with patch.object(controller, 'trained', side_effect=rows), \
             patch.object(controller, 'config_for'), patch.object(controller, 'read_record', return_value=tampered), \
             raises(RuntimeError, 'promotions'):
            controller.promotions_for(None, Path('/study'), {})

    def test_worker_rejects_direct_invocation_before_gpu_import(self):
        args = Mock(gpu='GPU-12345678', job='sports_to_clothing/c00/seed42', phase='train')
        with patch.dict(controller.os.environ, {}, clear=True), raises(RuntimeError, 'internal'):
            controller.worker(Path('/original'), Path('/study'), args)

    def test_scheduler_drains_sibling_on_controller_error(self):
        sibling, log = Mock(), Mock()
        def fail(*args):
            args[-1].append((sibling, {}, 'GPU-12345678', log))
            raise RuntimeError('identity changed')
        with patch.object(controller, '_schedule', side_effect=fail), raises(RuntimeError):
            controller.schedule(None, None, [], 'train', {}, {})
        sibling.wait.assert_called_once()
        log.close.assert_called_once()

    def test_manifest_hash_fails_before_source_or_original_import(self):
        with patch.object(controller, 'read', side_effect=[{'sha256': 'expected'}]), \
             patch.object(controller, 'sha', return_value='changed'), \
             patch.object(controller, 'original') as original, raises(RuntimeError, 'manifest'):
            controller.verify_study(Path('/active'), Path('/study'))
        original.assert_not_called()

    def test_completed_training_is_skipped_without_gpu_probe(self):
        with patch.object(controller, 'config_for', return_value=Mock()), \
             patch.object(controller, 'trained', return_value={'complete': True}), \
             patch.object(controller, 'gpu_snapshot') as gpu, \
             patch.object(controller.subprocess, 'Popen') as spawn:
            controller.schedule(None, None, controller.policy.screen_jobs(), 'train', {}, {'max_workers': 2})
        gpu.assert_not_called()
        spawn.assert_not_called()

    def test_failed_worker_stops_admissions_and_waits_sibling(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        first = Mock(returncode=1)
        first.poll.return_value = 1
        second = Mock(returncode=0)
        second.poll.side_effect = [None, 0]
        active = [(first, {}, 'GPU-first', Mock()), (second, {}, 'GPU-second', Mock())]
        with patch.object(controller.time, 'sleep'), patch.object(controller, 'seal'), \
             patch.object(controller, 'config_for', return_value=Mock()), \
             patch.object(controller, 'trained', return_value={'complete': True}), \
             patch.object(controller.subprocess, 'Popen') as spawn, raises(RuntimeError, 'admissions stopped'):
            controller._schedule(None, Path(temporary.name), [controller.policy.screen_jobs()[0]],
                                 'train', {}, {'max_workers': 2}, active)
        spawn.assert_not_called()
        assert active == []

    def test_resource_policy_requires_explicit_uuid_pool_and_budgets(self):
        args = Mock(gpus='GPU-12345678-1234-1234-1234-123456789abc,GPU-abcdef12-1234-1234-1234-123456789abc', max_workers=2, host_ram_gib=64., job_ram_gib=24.)
        assert controller.resource_policy(args)['max_workers'] == 2
        for field, value in [('gpus', '0,1'), ('max_workers', 3), ('host_ram_gib', 12), ('job_ram_gib', float('nan'))]:
            changed = Mock(**{**vars(args), field: value})
            with raises(ValueError):
                controller.resource_policy(changed)

    def test_immutable_receipts_reject_tampering(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        path = tmp_path / 'receipt.json'
        controller.seal(path, {'a': 1})
        controller.seal(path, {'a': 1})
        with raises(RuntimeError, match='Immutable'):
            controller.seal(path, {'a': 2})


    def test_os_lock_excludes_duplicate_workers(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        with controller.lock(tmp_path / 'worker.lock'):
            with raises(RuntimeError, match='locked'):
                with controller.lock(tmp_path / 'worker.lock'):
                    pass


    def test_gpu_guard_requires_idle_uuid_and_memory(self):
        cards = [{'uuid': 'GPU-abc', 'free_mib': 13000, 'utilization': 0}]
        controller.admit('GPU-abc', cards, set(), 24, 48, 24, 48)
        for busy, free, available in [(set(['GPU-abc']), 13000, 30), (set(), 12000, 30), (set(), 13000, 23)]:
            with raises(RuntimeError):
                controller.admit('GPU-abc', [{'uuid': 'GPU-abc', 'free_mib': free, 'utilization': 0}], busy, 24, 48, 24, available)
        with raises(RuntimeError):
            controller.admit('GPU-abc', cards, set(), 48, 48, 24, 30)


    def test_training_completion_binds_full_identity(self):
        identity = {'selection_policy': {'metric': 'NDCG@10'}}
        state = {'identity': identity, 'epoch': 2, 'validation': {'N@10': .4}}
        manifest = {'identity': identity, 'complete': True, 'epochs_completed': 3, 'best_validation': state['validation']}
        controller.check_training(manifest, state, identity)
        for changed in [dict(manifest, complete=False), dict(manifest, best_validation={}), dict(manifest, identity={})]:
            with raises(RuntimeError):
                controller.check_training(changed, state, identity)
        with raises(RuntimeError):
            controller.check_training(manifest, dict(state, identity={}), identity)


    def test_verification_requires_every_binding_and_flag(self):
        identity = {'checkpoint_sha256': 'a'}
        hashes = {'snapshot_sha256': 's', 'cache_sha256': 'c', 'search_sha256': 'h'}
        search = {'identity': identity, 'complete': True, 'split': 'validation', **hashes}
        stored = {'identity': identity, 'complete': True, 'snapshot_sha256': 's'}
        verification = {'binding': {'identity': identity, **hashes}, 'raw_scores_equal': True,
                        'rankings_equal': True, 'verified_all_metrics': True, 'metric_differences': {'N@10': 0}}
        controller.check_verification(search, stored, verification, identity, hashes)
        for flag in ('raw_scores_equal', 'rankings_equal', 'verified_all_metrics'):
            with raises(RuntimeError):
                controller.check_verification(search, stored, dict(verification, **{flag: False}), identity, hashes)
        with raises(RuntimeError):
            controller.check_verification(search, dict(stored, complete=False), verification, identity, hashes)
        with raises(RuntimeError):
            controller.check_verification(search, stored, verification, identity, dict(hashes, cache_sha256='tampered'))


    def test_paths_fail_closed(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        for name in ('../escape', '/absolute', 'a/b', ''):
            with raises(ValueError):
                controller.study_path(tmp_path, name)
        assert controller.study_path(tmp_path, 'trial-01') == tmp_path / 'artifacts/hparam_search/trial-01'


    def test_plan_has_no_test_action(self):
        plan = controller.plan()
        assert plan['new_training_budget'] == 24
        assert plan['validation_models'] == 12
        assert len(plan['screen_jobs']) == 16
        assert len(plan['fusion_settings']) == 48
        with raises(SystemExit):
            controller.parser().parse_args(['--action', 'test'])
