import json
from pathlib import Path
import tempfile
import unittest

from cdr_framework import experiment_control as ec


class ControlTests(unittest.TestCase):
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
