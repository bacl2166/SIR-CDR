import ctypes
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments import extend_gencdr_run as extension


class PidfdCompatTests(unittest.TestCase):
    def test_conda_without_python_interfaces_uses_typed_libc_pidfd(self):
        libc = Mock()
        libc.pidfd_open.return_value = 99
        libc.pidfd_send_signal.return_value = 0
        process = {'pid': 123, 'uid': 1000, 'start_ticks': 777, 'cwd': '/repo',
                   'argv': ['python', 'experiments/run_gencdr_six.py', '--action',
                            'run-validation', '--batch-id', 'batch']}
        with patch.object(os, 'pidfd_open', None, create=True), \
                patch.object(signal, 'pidfd_send_signal', None, create=True), \
                patch.object(sys, 'platform', 'linux'), \
                patch('ctypes.CDLL', return_value=libc) as loaded, \
                patch.object(os, 'getuid', return_value=1000, create=True), \
                patch.object(os, 'close') as closed, \
                patch.object(extension, 'process_snapshot', side_effect=[process, None]):
            extension.terminate_controller(process, Path('/repo'), 'batch')
        loaded.assert_called_once_with(None, use_errno=True)
        libc.pidfd_open.assert_called_once_with(123, 0)
        libc.pidfd_send_signal.assert_called_once_with(99, signal.SIGTERM, None, 0)
        self.assertEqual(libc.pidfd_open.argtypes, [ctypes.c_int, ctypes.c_uint])
        self.assertEqual(libc.pidfd_send_signal.argtypes,
                         [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint])
        closed.assert_called_once_with(99)

    def test_libc_errno_is_propagated_without_numeric_pid_fallback(self):
        libc = Mock()
        libc.pidfd_open.return_value = -1
        with patch.object(os, 'pidfd_open', None, create=True), \
                patch.object(signal, 'pidfd_send_signal', None, create=True), \
                patch.object(sys, 'platform', 'linux'), \
                patch('ctypes.CDLL', return_value=libc), \
                patch('ctypes.get_errno', return_value=errno.EPERM), \
                patch.object(os, 'kill') as killed:
            with self.assertRaises(OSError) as raised:
                extension.PidfdBackend().open(123)
            self.assertEqual(raised.exception.errno, errno.EPERM)
            killed.assert_not_called()

    def test_libc_rejects_reused_identity_without_sending_signal(self):
        libc = Mock()
        libc.pidfd_open.return_value = 99
        with patch.object(os, 'pidfd_open', None, create=True), \
                patch.object(signal, 'pidfd_send_signal', None, create=True), \
                patch.object(sys, 'platform', 'linux'), \
                patch('ctypes.CDLL', return_value=libc), \
                patch.object(os, 'close') as closed, \
                patch.object(extension, 'process_snapshot', return_value={'start_ticks': 2}):
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                extension.terminate_controller({'pid': 123, 'start_ticks': 1}, Path('/repo'), 'batch')
            libc.pidfd_send_signal.assert_not_called()
            closed.assert_called_once_with(99)

    def test_libc_send_failure_propagates_errno_and_closes_descriptor(self):
        libc = Mock()
        libc.pidfd_open.return_value = 99
        libc.pidfd_send_signal.return_value = -1
        process = {'pid': 123, 'uid': 1000, 'start_ticks': 777, 'cwd': '/repo',
                   'argv': ['python', 'experiments/run_gencdr_six.py', '--action',
                            'run-validation', '--batch-id', 'batch']}
        with patch.object(os, 'pidfd_open', None, create=True), \
                patch.object(signal, 'pidfd_send_signal', None, create=True), \
                patch.object(sys, 'platform', 'linux'), \
                patch('ctypes.CDLL', return_value=libc), \
                patch('ctypes.get_errno', return_value=errno.EPERM), \
                patch.object(os, 'getuid', return_value=1000, create=True), \
                patch.object(os, 'close') as closed, \
                patch.object(extension, 'process_snapshot', return_value=process):
            with self.assertRaises(OSError) as raised:
                extension.terminate_controller(process, Path('/repo'), 'batch')
            self.assertEqual(raised.exception.errno, errno.EPERM)
            closed.assert_called_once_with(99)

    def test_self_probe_uses_signal_zero_and_closes_descriptor_on_failure(self):
        backend = Mock()
        backend.name = 'test'
        backend.open.return_value = 99
        backend.send.side_effect = PermissionError('blocked')
        with patch.object(os, 'close') as closed:
            with self.assertRaises(PermissionError):
                extension.PidfdBackend.probe(backend)
        backend.open.assert_called_once_with(os.getpid())
        backend.send.assert_called_once_with(99, 0)
        closed.assert_called_once_with(99)

    def test_missing_libc_symbols_stop_without_signalling(self):
        with patch.object(os, 'pidfd_open', None, create=True), \
                patch.object(signal, 'pidfd_send_signal', None, create=True), \
                patch.object(sys, 'platform', 'linux'), \
                patch('ctypes.CDLL', return_value=object()), \
                patch.object(os, 'kill') as killed:
            with self.assertRaisesRegex(RuntimeError, 'pidfd'):
                extension.PidfdBackend()
            killed.assert_not_called()

    @unittest.skipUnless(sys.platform == 'linux', 'real Linux pidfd integration')
    def test_real_libc_signals_only_selected_child_without_python_interfaces(self):
        # Harmless children, never an experiment process. Also runs on the server.
        with subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']) as target, \
                subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']) as sibling:
            try:
                with patch.object(os, 'pidfd_open', None, create=True), \
                        patch.object(signal, 'pidfd_send_signal', None, create=True):
                    backend = extension.PidfdBackend()
                    self.assertEqual(backend.probe()['backend'], 'glibc')
                    descriptor = backend.open(target.pid)
                    try:
                        backend.send(descriptor, signal.SIGTERM)
                    finally:
                        os.close(descriptor)
                    self.assertEqual(target.wait(timeout=5), -signal.SIGTERM)
                    self.assertIsNone(sibling.poll())
            finally:
                for child in (target, sibling):
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=5)


class RecoveryBindingTests(unittest.TestCase):
    def values(self):
        old = {'coordinator_commit': 'd857f1cdb5b8bf07296db98de7eb6e4e8e56c0d8',
               'coordinator_sha256': '3205658dad424a15f78e62351a1b5e27c10cb1994c4ff289927338f731834814',
               'gpus': [4, 5, 6, 7], 'maximum_concurrency': 4}
        new = old | {'coordinator_commit': 'new-commit', 'coordinator_sha256': 'new-sha'}
        return old, new

    def coordinator(self, folder):
        coordinator = extension.Coordinator.__new__(extension.Coordinator)
        coordinator.control = Path(folder)
        coordinator.resource = coordinator.control / 'resource_extension.json'
        coordinator.resource_value = self.values()[1]
        return coordinator

    def test_failed_legacy_handoff_preserves_resource_and_adds_bound_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.coordinator(folder)
            old, new = self.values()
            extension.immutable_receipt(coordinator.resource, old)
            original = coordinator.resource.read_bytes()
            coordinator.bind_resource()
            self.assertEqual(coordinator.resource_value, old)
            self.assertEqual(coordinator.resource.read_bytes(), original)
            self.assertFalse(coordinator.repair_path.exists())  # inspect is read-only
            self.assertEqual(coordinator.repair_value['resource_sha256'], extension.sha(coordinator.resource))
            self.assertEqual(coordinator.repair_value['replacement_coordinator_commit'], new['coordinator_commit'])
            extension.immutable_receipt(coordinator.repair_path, coordinator.repair_value)
            coordinator.verify_repair()
            coordinator.repair_path.write_text('{}')
            with self.assertRaises(RuntimeError):
                coordinator.verify_repair()

    def test_resource_change_or_unknown_legacy_code_cannot_use_compat_recovery(self):
        for change in ({'maximum_concurrency': 6}, {'coordinator_sha256': 'unknown'}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                coordinator = self.coordinator(folder)
                old, _ = self.values()
                extension.immutable_receipt(coordinator.resource, old | change)
                with self.assertRaises(RuntimeError):
                    coordinator.bind_resource()

    def test_new_repair_is_closed_after_handoff_or_freeze_but_existing_can_resume(self):
        for marker in ('extension_handoff_complete.json', 'validation_freeze.json'):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as folder:
                coordinator = self.coordinator(folder)
                old, _ = self.values()
                extension.immutable_receipt(coordinator.resource, old)
                coordinator.bind_resource()
                repair = coordinator.repair_value
                (coordinator.control / marker).write_text('{}')
                with self.assertRaises(RuntimeError):
                    self.coordinator(folder).bind_resource()
                extension.immutable_receipt(coordinator.repair_path, repair)
                self.coordinator(folder).bind_resource()

    def test_repair_hash_is_required_by_freeze_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.coordinator(folder)
            old, _ = self.values()
            extension.immutable_receipt(coordinator.resource, old)
            coordinator.bind_resource()
            extension.immutable_receipt(coordinator.repair_path, coordinator.repair_value)
            freeze = coordinator.control / 'validation_freeze.json'
            freeze.write_text('{}')
            receipt = {'resource_sha256': extension.sha(coordinator.resource),
                       'validation_freeze_sha256': extension.sha(freeze)}
            path = coordinator.control / 'extension_freeze.json'
            path.write_text(json.dumps(receipt))
            coordinator.pre_admission = Mock()
            coordinator.root, coordinator.batch = coordinator.control, 'batch'
            coordinator.core = Mock()
            with self.assertRaisesRegex(RuntimeError, 'recovery differs'):
                coordinator.guard_freeze()
            path.write_text(json.dumps(receipt | coordinator.recovery_binding()))
            coordinator.guard_freeze()


if __name__ == '__main__':
    unittest.main()
