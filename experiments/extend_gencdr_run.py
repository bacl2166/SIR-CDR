"""Separately versioned resource coordinator; never updates the live training checkout.

Run this file from a separate clean control checkout, with --root pointing to the
pinned, already-running training checkout. Only Linux supports process handoff.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


TRAINING_COMMIT = 'a887bd389ff72eb14ec96be0dfb876a61dae9331'
LEGACY_COORDINATOR = {
    'coordinator_commit': 'd857f1cdb5b8bf07296db98de7eb6e4e8e56c0d8',
    'coordinator_sha256': '3205658dad424a15f78e62351a1b5e27c10cb1994c4ff289927338f731834814',
}


class PidfdBackend:
    """Use a stable pidfd even when Conda Python omits its Linux wrappers."""
    def __init__(self):
        self._open = getattr(os, 'pidfd_open', None)
        self._send = getattr(signal, 'pidfd_send_signal', None)
        if callable(self._open) and callable(self._send):
            self.name = 'python'
            return
        if sys.platform != 'linux':
            raise RuntimeError('pidfd requires Linux; no numeric-PID fallback')
        try:
            self._libc = ctypes.CDLL(None, use_errno=True)
            self._open = self._libc.pidfd_open
            self._send = self._libc.pidfd_send_signal
        except (AttributeError, OSError) as error:
            raise RuntimeError('pidfd unavailable in Python and libc; no numeric-PID fallback') from error
        self._open.argtypes = [ctypes.c_int, ctypes.c_uint]
        self._open.restype = ctypes.c_int
        self._send.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        self._send.restype = ctypes.c_int
        self.name = 'glibc'

    @staticmethod
    def checked(result):
        if result == -1:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return result

    def open(self, pid):
        if self.name == 'python':
            return self._open(pid)
        return self.checked(self._open(pid, 0))

    def send(self, descriptor, signum):
        if self.name == 'python':
            return self._send(descriptor, signum)
        return self.checked(self._send(descriptor, signum, None, 0))

    def probe(self):
        """Signal zero checks only our own process; it delivers no signal."""
        descriptor = self.open(os.getpid())
        try:
            self.send(descriptor, 0)
        finally:
            os.close(descriptor)
        return {'backend': self.name, 'self_probe': 'passed'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable_receipt(path, payload):
    """Exclusive creation: an existing receipt is checked, never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read(path) != payload:
                raise RuntimeError(f'Immutable receipt differs: {path}')
    finally:
        Path(temporary).unlink(missing_ok=True)
    return payload


def eligible_jobs(pairs, seeds, completed, active):
    known = {(p, s) for p in pairs for s in seeds}
    if (set(completed) | set(active)) - known:
        raise RuntimeError('Unknown job in queue state')
    pending = [(p, s) for s in seeds for p in pairs if (p, s) not in completed and (p, s) not in active]
    primary = []
    for pair in pairs:
        if any(p == pair for p, _ in active):
            continue
        candidate = next((j for j in pending if j[0] == pair), None)
        if candidate:
            primary.append(candidate)
    key = lambda j: (seeds.index(j[1]), pairs.index(j[0]))
    # Prefer six different directions. Once a direction has no remaining work,
    # independent seeds of other directions may fill otherwise empty slots.
    return sorted(primary, key=key) + sorted((j for j in pending if j not in primary), key=key)


def option(argv, name):
    try:
        index = argv.index(name)
        return argv[index + 1]
    except (ValueError, IndexError):
        return None


def script_matches(process, root, name):
    target = (Path(root) / 'experiments' / name).resolve()
    for value in process['argv'][1:]:
        if value.endswith('.py'):
            path = Path(value)
            return (path if path.is_absolute() else Path(process['cwd']) / path).resolve() == target
    return False


def validate_controller(process, root, batch, uid):
    if (process['uid'] != uid or Path(process['cwd']).resolve() != Path(root).resolve()
            or not script_matches(process, root, 'run_gencdr_six.py')
            or option(process['argv'], '--action') != 'run-validation'
            or option(process['argv'], '--batch-id') != batch):
        raise RuntimeError('Controller owner/root/script/action/batch identity mismatch')


def identify_job(process, root, batch, configs):
    if Path(process['cwd']).resolve() != Path(root).resolve():
        return None
    argv = process['argv']
    if script_matches(process, root, 'run_gencdr_shared.py'):
        if (option(argv, '--batch-id') == batch
                and option(argv, '--action') in ('train', 'validate', 'test')):
            pair, seed = option(argv, '--pairs'), option(argv, '--seed')
            if pair is not None and seed is not None:
                return pair, int(seed)
    if any(script_matches(process, root, name) for name in
           ('run_llm_cdr.py', 'recover_llm_evaluation.py')):
        path = option(argv, '--config')
        if path is not None:
            path = Path(path)
            resolved = str((path if path.is_absolute() else Path(process['cwd']) / path).resolve())
            return configs.get(resolved)
    return None


def check_training_completion(manifest, state, identity):
    if (manifest.get('complete') is not True or manifest.get('identity') != identity
            or state.get('identity') != identity
            or not 0 < state.get('epoch', 0) <= manifest.get('epochs_completed', 0)
            or manifest.get('best_validation') != state.get('validation')):
        raise RuntimeError('Training completion/checkpoint identity is inconsistent')


def verify_extension_freeze(receipt, resource_hash, freeze_hash):
    if (receipt.get('resource_sha256') != resource_hash
            or receipt.get('validation_freeze_sha256') != freeze_hash):
        raise RuntimeError('Resource extension or canonical freeze changed')


def process_snapshot(pid, proc=Path('/proc')):
    folder = Path(proc) / str(pid)
    try:
        fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'uid': folder.stat().st_uid, 'ppid': int(fields[1]),
                'start_ticks': int(fields[19]), 'state': fields[0],
                'cwd': str((folder / 'cwd').resolve(strict=True)),
                'argv': (folder / 'cmdline').read_bytes().decode('utf-8', 'replace').strip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def owned_processes():
    if sys.platform != 'linux':
        raise RuntimeError('Live handoff requires Linux /proc and pidfd support')
    result = []
    for folder in Path('/proc').iterdir():
        if folder.name.isdigit():
            try:
                if folder.stat().st_uid != os.getuid():
                    continue
            except FileNotFoundError:
                continue
            value = process_snapshot(int(folder.name))
            if value:
                result.append(value)
    return result


def expected_uuid(process):
    """Read only the GPU binding keys from own-worker environment; never log secrets."""
    raw = (Path('/proc') / str(process['pid']) / 'environ').read_bytes().split(b'\0')
    values = {}
    for entry in raw:
        key, _, value = entry.partition(b'=')
        if key in (b'CUDA_VISIBLE_DEVICES', b'SIR_EXPECTED_GPU_UUID'):
            values[key.decode()] = value.decode()
    uuid = values.get('SIR_EXPECTED_GPU_UUID')
    if not uuid or values.get('CUDA_VISIBLE_DEVICES') != uuid or not uuid.startswith('GPU-'):
        raise RuntimeError('Own worker has an ambiguous GPU binding')
    return uuid


def terminate_controller(process, root, batch):
    """Signal exactly one stable own controller, never a PID group or its workers."""
    backend = PidfdBackend()
    descriptor = backend.open(process['pid'])
    try:
        current = process_snapshot(process['pid'])
        if current is None or current['start_ticks'] != process['start_ticks']:
            raise RuntimeError('Controller exited or PID identity changed before handoff')
        validate_controller(current, root, batch, os.getuid())
        backend.send(descriptor, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            current = process_snapshot(process['pid'])
            if current is None or current['start_ticks'] != process['start_ticks']:
                return
            time.sleep(.1)
        raise RuntimeError('Controller did not exit; workers were not signalled')
    finally:
        os.close(descriptor)


class Coordinator:
    def __init__(self, root, batch, gpus, maximum, poll):
        self.root, self.batch = Path(root).resolve(), batch
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', batch):
            raise ValueError('Invalid batch ID')
        source = Path(__file__).resolve().parents[1]
        if source == self.root:
            raise RuntimeError('Use a separate control checkout; keep live training checkout unchanged')
        self.control = self.root / 'artifacts/experiment_control/gencdr-shared-loo-v1' / batch
        self.declaration = read(self.control / 'experiment_manifest.json')
        self.assert_git(self.root, TRAINING_COMMIT)
        if self.declaration.get('commit') != TRAINING_COMMIT:
            raise RuntimeError('Unsupported batch training commit')
        control_commit = self.git(source, 'rev-parse', 'HEAD')
        self.assert_git(source, control_commit)
        sys.path.insert(0, str(self.root))
        self.core = importlib.import_module('cdr_framework.experiment_control')
        self.runner = importlib.import_module('experiments.run_gencdr_six')
        self.shared = importlib.import_module('experiments.run_gencdr_shared')
        self.recovery = importlib.import_module('experiments.recover_llm_evaluation')
        self.tuning = importlib.import_module('experiments.tune_llm_fusion')
        for name, module in list(sys.modules.items()):
            if name.startswith(('cdr_framework', 'experiments.')) and getattr(module, '__file__', None):
                if not Path(module.__file__).resolve().is_relative_to(self.root):
                    raise RuntimeError(f'Import escaped pinned training checkout: {name}')
        self.pairs, self.seeds = self.core.SIX_PAIRS, self.core.SEEDS
        if self.declaration.get('pairs') != list(self.pairs) or self.declaration.get('seeds') != list(self.seeds):
            raise RuntimeError('Original declaration does not cover all18 jobs')
        self.gpus = self.core.parse_gpu_ids(gpus)
        if not set(self.declaration['gpus']).issubset(self.gpus) or not 1 <= maximum <= len(self.gpus):
            raise RuntimeError('Keep all original affinity cards and supply a valid concurrency cap')
        self.maximum, self.poll = maximum, poll
        self.configs = {(p, s): self.shared.model_config(self.root, p, s, batch)
                        for p in self.pairs for s in self.seeds}
        self.runtime_map = {str(self.shared.runtime_path(self.root, p, s, batch).resolve()): (p, s)
                            for p, s in self.configs}
        cards = self.core.query_gpu_inventory()
        selected = [c for c in cards if c['index'] in self.gpus]
        if len(selected) != len(self.gpus):
            raise RuntimeError('A requested GPU is absent')
        self.by_uuid = {c['uuid']: c['index'] for c in selected}
        self.resource = self.control / 'resource_extension.json'
        self.resource_value = {'schema': 'six-direction-resource-extension-v1', 'batch_id': batch,
            'original_manifest_sha256': sha(self.control / 'experiment_manifest.json'),
            'training_commit': TRAINING_COMMIT, 'coordinator_commit': control_commit,
            'coordinator_sha256': sha(__file__), 'maximum_concurrency': maximum,
            'gpus': [{'index': c['index'], 'uuid': c['uuid']} for c in selected],
            'seed_policy': 'prefer uncovered directions;independent fixed seeds42/43/44;global18-job freeze',
            'authorization': 'user authorized remaining idle GPUs and later seeds;never repeat completed jobs'}
        self.bind_resource()
        for pair in self.pairs:
            smoke = read(self.control / 'smoke' / f'{pair}.json')
            if not smoke.get('passed') or smoke['identity'] != self.runner.smoke_identity(self.root, batch, pair):
                raise RuntimeError('Missing or stale original smoke gate')

    def bind_resource(self):
        """Append provenance for the known pre-handoff compatibility failure only."""
        current = dict(self.resource_value)
        self.coordinator_sha256 = current['coordinator_sha256']
        self.repair_path = self.control / 'extension_pidfd_recovery.json'
        self.repair_value = None
        if not self.resource.exists() or read(self.resource) == current:
            if self.repair_path.exists():
                raise RuntimeError('Unexpected pidfd recovery receipt')
            return
        old = read(self.resource)
        if (any(old.get(key) != value for key, value in LEGACY_COORDINATOR.items())
                or {k: v for k, v in old.items() if k not in LEGACY_COORDINATOR}
                != {k: v for k, v in current.items() if k not in LEGACY_COORDINATOR}):
            raise RuntimeError('Existing resource amendment differs; do not overwrite it')
        self.repair_value = {
            'schema': 'pidfd-compatibility-recovery-v1',
            'resource_sha256': sha(self.resource),
            'previous_coordinator_commit': old['coordinator_commit'],
            'previous_coordinator_sha256': old['coordinator_sha256'],
            'replacement_coordinator_commit': current['coordinator_commit'],
            'replacement_coordinator_sha256': current['coordinator_sha256'],
            'reason': 'Conda Python omits pidfd wrappers; use stable libc pidfd without changing experiment',
        }
        if self.repair_path.exists():
            self.verify_repair()
        elif any((self.control / name).exists() for name in
                 ('extension_handoff_complete.json', 'validation_freeze.json')):
            raise RuntimeError('New pidfd recovery is closed after handoff/freeze')
        self.resource_value = old

    def verify_repair(self):
        if self.repair_value is not None and (
                not self.repair_path.exists() or read(self.repair_path) != self.repair_value):
            raise RuntimeError('Pidfd recovery receipt missing or changed')

    def recovery_binding(self):
        self.verify_repair()
        return ({'coordinator_recovery_sha256': sha(self.repair_path)}
                if self.repair_value is not None else {})

    @staticmethod
    def git(root, *arguments):
        return subprocess.run(['git', '-C', str(root), *arguments], check=True,
                              text=True, capture_output=True, timeout=30).stdout.strip()

    @classmethod
    def assert_git(cls, root, commit):
        if cls.git(root, 'rev-parse', 'HEAD') != commit or cls.git(root, 'status', '--porcelain'):
            raise RuntimeError(f'Pinned checkout changed or is dirty: {root}')

    def inventory(self):
        return self.core.query_gpu_inventory()

    def live(self):
        result = {}
        for process in owned_processes():
            job = identify_job(process, self.root, self.batch, self.runtime_map)
            if job is None:
                continue
            if job not in self.configs:
                raise RuntimeError('Own batch process has an unknown job identity')
            try:
                uuid = expected_uuid(process)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if uuid not in self.by_uuid:
                raise RuntimeError('Live worker GPU is outside amended pool')
            result.setdefault(job, []).append(process | {'gpu_uuid': uuid})
        return result

    def controller(self):
        values = [p for p in owned_processes()
                  if script_matches(p, self.root, 'run_gencdr_six.py')
                  and option(p['argv'], '--batch-id') == self.batch]
        if len(values) != 1:
            raise RuntimeError('Initial takeover requires exactly one original controller')
        validate_controller(values[0], self.root, self.batch, os.getuid())
        return values[0]

    def affinity(self, job):
        path = self.control / 'affinity' / f'{job[0]}_seed{job[1]}.json'
        return read(path)['gpu_uuid'] if path.exists() else None

    def completed(self, job):
        pair, seed = job
        cfg = self.configs[job]
        tuning = self.root / 'artifacts/tuning' / self.core.SCHEMA / self.batch / pair / f'seed{seed}'
        if all(p.exists() for p in (cfg.output_dir / 'manifest.json', tuning / 'search.json', tuning / 'verification.json')):
            return self.core.collect_validation_binding(self.root, self.batch, pair, seed,
                                                        allow_test=(self.control / 'validation_freeze.json').exists())
        return None

    def write_status(self, phase, job, **extra):
        path = self.control / 'status' / phase / f'{job[0]}_seed{job[1]}.json'
        old = read(path) if path.exists() else {}
        value = old | {'pair': job[0], 'seed': job[1], 'extension': True} | extra
        self.core.write_json_atomic(path, value)

    def inspect(self):
        pidfd = PidfdBackend().probe()
        live = self.live()
        done = [j for j in self.configs if j not in live and self.completed(j)]
        snapshot = self.core.capture_environment(self.root, Path(self.declaration['data_root']), self.gpus)
        return {'pidfd': pidfd, 'pidfd_recovery_plan': self.repair_value,
                'resource_plan': self.resource_value, 'completed_validation_jobs': [list(j) for j in done],
                'live_jobs': [{'pair': j[0], 'seed': j[1], 'gpu_uuid': rows[0]['gpu_uuid'],
                               'pids': [r['pid'] for r in rows]} for j, rows in live.items()],
                'admission': snapshot['admission'], 'gpu_inventory': snapshot['gpu_inventory']}

    def validate_adoption(self, live):
        used = set()
        if len(live) > self.maximum:
            raise RuntimeError('More live jobs than the approved concurrency cap')
        for job, rows in live.items():
            uuids = {p['gpu_uuid'] for p in rows}
            if len(uuids) != 1 or self.affinity(job) not in uuids:
                raise RuntimeError('Adopted job GPU affinity differs')
            uuid = next(iter(uuids))
            if uuid in used:
                raise RuntimeError('Ambiguous adopted GPU ownership')
            card = next(c for c in self.inventory() if c['uuid'] == uuid)
            known = {p['pid'] for p in rows}
            chain = {p['pid']: p for p in rows}
            for pid in set(card['pids']) - known:
                # A matching inner worker may have started after the first scan.
                process = process_snapshot(pid)
                if (process is None or process['uid'] != os.getuid()
                        or identify_job(process, self.root, self.batch, self.runtime_map) != job
                        or expected_uuid(process) != uuid):
                    raise RuntimeError('An adopted card contains an unrelated process')
                chain[pid] = process
            roots = [p for p in chain if chain[p]['ppid'] not in chain]
            children = {p: [child for child in chain if chain[child]['ppid'] == p] for p in chain}
            if len(roots) != 1 or any(len(values) > 1 for values in children.values()):
                raise RuntimeError('Ambiguous duplicate worker chains for one direction/seed')
            visited, current = set(), roots[0]
            while current not in visited:
                visited.add(current)
                if not children[current]:
                    break
                current = children[current][0]
            if visited != set(chain):
                raise RuntimeError('Disconnected or cyclic worker ownership')
            used.add(uuid)

    def pre_admission(self):
        self.assert_git(self.root, TRAINING_COMMIT)
        if read(self.resource) != self.resource_value or sha(__file__) != self.coordinator_sha256:
            raise RuntimeError('Coordinator/resource amendment changed')
        self.verify_repair()
        self.runner.check_host_pressure()
        self.runner.check_stage_storage(self.root, self.batch, 'run-validation')

    def handoff(self):
        marker = self.control / 'extension_handoff_complete.json'
        if marker.exists():
            if read(marker) != {'resource_sha256': sha(self.resource)}:
                raise RuntimeError('Handoff receipt differs')
            return
        if (self.control / 'validation_freeze.json').exists() or any(
                (cfg.output_dir / name).exists() for cfg in self.configs.values() for name in ('test.json', 'test.lock')):
            raise RuntimeError('Initial resource takeover is closed after freeze/test access')
        intent = self.control / 'extension_handoff_intent.json'
        if self.resource.exists() and intent.exists():
            previous = read(intent)
            if previous['resource_sha256'] != sha(self.resource):
                raise RuntimeError('Handoff intent resource mismatch')
            old = process_snapshot(previous['controller']['pid'])
            if old is None or old['start_ticks'] != previous['controller']['start_ticks']:
                # Recover a helper crash after controller exit but before the
                # handoff-complete receipt. The old immutable intent is retained.
                return
        controller = self.controller()
        snapshot = self.core.capture_environment(self.root, Path(self.declaration['data_root']), self.gpus)
        if snapshot['admission']['errors']:
            raise RuntimeError(str(snapshot['admission']))
        added = set(self.gpus) - set(self.declaration['gpus'])
        if not added.issubset(snapshot['admission']['allowed_gpus']):
            raise RuntimeError('A newly approved card is no longer idle; original controller left running')
        self.validate_adoption(self.live())
        immutable_receipt(self.resource, self.resource_value)
        if not intent.exists():
            immutable_receipt(intent, {'controller': controller,
                'resource_sha256': sha(self.resource), 'time': self.core.utc_now()})
        elif read(intent)['resource_sha256'] != sha(self.resource):
            raise RuntimeError('Handoff intent resource mismatch')
        terminate_controller(controller, self.root, self.batch)
        print(json.dumps({'handoff': 'original controller exited; training workers were not signalled',
                          'controller_pid': controller['pid']}), flush=True)

    def wait_existing(self, job):
        while self.live().get(job):
            time.sleep(min(self.poll, 5))
        # The matching per-job OS lock also guards a wrapper finishing after its
        # inner process exits. No state is inferred from PID exit alone.
        with self.recovery.operation_lock(self.control / 'locks' / f'{job[0]}_seed{job[1]}'):
            pass

    def stage(self, job, action, uuid, log):
        self.pre_admission()
        if action == 'test':
            self.guard_freeze()
        card = next(c for c in self.inventory() if c['uuid'] == uuid)
        minimum = self.runner.training_memory_requirement(read(self.control / 'smoke' / f'{job[0]}.json')['report'])
        self.core.assert_gpu_launchable({'requested_gpu_whitelist': self.gpus,
                                        'gpu_inventory': self.inventory()}, card['index'], minimum)
        command = [sys.executable, str(self.root / 'experiments/run_gencdr_shared.py'),
                   '--action', action, '--root', str(self.root), '--pairs', job[0],
                   '--seed', str(job[1]), '--batch-id', self.batch, '--device', 'cuda:0']
        # Write-ahead state plus exact config-based /proc discovery handles a
        # coordinator crash between Popen and the PID log write.
        self.write_status('run-test' if action == 'test' else 'run-validation', job,
                          state='running', gpu=card['index'], gpu_uuid=uuid, stage=action,
                          launch_intent_at=self.core.utc_now())
        self.runner.run_commands([command], self.root, log, uuid)

    def work(self, job, uuid, adopted=False, testing=False):
        if adopted:
            self.wait_existing(job)
        pair, seed = job
        log = self.root / 'logs' / self.core.SCHEMA / self.batch / f'{pair}_seed{seed}_extension.log'
        if testing:
            frozen = self.guard_freeze()
            binding = next(j for j in frozen['jobs'] if (j['pair'], j['seed']) == job)
            if (self.configs[job].output_dir / 'test.json').exists():
                self.core.verify_test_completion(self.root, self.batch, binding)
                return
            self.stage(job, 'test', uuid, log)
            self.core.verify_test_completion(self.root, self.batch, binding)
            return
        if self.completed(job):
            return
        cfg = self.configs[job]
        manifest = cfg.output_dir / 'manifest.json'
        if manifest.exists():
            state, identity = self.tuning.validate_checkpoint(cfg, cfg.output_dir / 'best_model.pt')
            check_training_completion(read(manifest), state, identity)
        else:
            self.stage(job, 'train', uuid, log)
        self.stage(job, 'validate', uuid, log)
        if not self.completed(job):
            raise RuntimeError('Job returned without a canonical exact validation binding')

    def queue(self, testing=False):
        live = self.live()
        self.validate_adoption(live)
        done = set()
        bindings = self.guard_freeze()['jobs'] if testing else []
        for job in self.configs:
            if job in live:
                continue
            if testing:
                binding = next(j for j in bindings if (j['pair'], j['seed']) == job)
                if (self.configs[job].output_dir / 'test.json').exists():
                    self.core.verify_test_completion(self.root, self.batch, binding)
                    done.add(job)
            elif self.completed(job):
                done.add(job)
        active, used = {}, set()
        failure = None
        phase = 'run-test' if testing else 'run-validation'
        with ThreadPoolExecutor(max_workers=self.maximum) as pool:
            for job, rows in live.items():
                uuids = {p['gpu_uuid'] for p in rows}
                if len(uuids) != 1 or self.affinity(job) not in uuids:
                    raise RuntimeError('Adopted job GPU affinity differs')
                uuid = uuids.pop()
                if uuid in used:
                    raise RuntimeError('Ambiguous adopted GPU ownership')
                if len(active) >= self.maximum:
                    raise RuntimeError('More live jobs than the approved concurrency cap')
                active[pool.submit(self.work, job, uuid, True, testing)] = job
                used.add(uuid)
                self.write_status(phase, job, state='running', adopted=True, gpu=self.by_uuid[uuid])
            while active or len(done) < len(self.configs):
                if failure is None:
                    candidates = ([j for j in self.configs if j not in done and j not in active.values()]
                                  if testing else eligible_jobs(self.pairs, self.seeds, done, set(active.values())))
                    for job in candidates:
                        if len(active) >= self.maximum:
                            break
                        self.pre_admission()
                        cards = self.inventory()
                        allowed = self.core.admit_gpu_whitelist(self.gpus, cards)['allowed_gpus']
                        expected = self.affinity(job)
                        if testing:
                            expected = next(j['gpu_uuid'] for j in bindings if (j['pair'], j['seed']) == job)
                        minimum = self.runner.training_memory_requirement(read(self.control / 'smoke' / f'{job[0]}.json')['report'])
                        card = next((c for c in cards if c['index'] in allowed and c['uuid'] not in used
                                     and c['memory_free_mib'] >= minimum
                                     and (not expected or c['uuid'] == expected)), None)
                        if card is None:
                            continue
                        self.runner.admit_host_budget(
                            available=self.core.host_resources()['memory'].get('MemAvailable', 0),
                            reserved=len(active) * 12 * 2**30, required=12 * 2**30)
                        self.core.assert_gpu_launchable({'requested_gpu_whitelist': self.gpus,
                            'gpu_inventory': self.inventory()}, card['index'], minimum)
                        affinity = self.control / 'affinity' / f'{job[0]}_seed{job[1]}.json'
                        immutable_receipt(affinity, {'pair': job[0], 'seed': job[1], 'gpu_uuid': card['uuid']})
                        self.write_status(phase, job, state='running', gpu=card['index'], adopted=False,
                                          started_at=self.core.utc_now())
                        used.add(card['uuid'])
                        active[pool.submit(self.work, job, card['uuid'], False, testing)] = job
                        print(json.dumps({'admitted': job, 'gpu': card['index'], 'gpu_uuid': card['uuid']}), flush=True)
                if not active:
                    if len(done) < len(self.configs):
                        raise RuntimeError(failure or 'Pending jobs have no idle approved affinity card; rerun to resume')
                    break
                finished, _ = wait(active, timeout=self.poll, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = active.pop(future)
                    used.discard(self.affinity(job))
                    try:
                        future.result()
                        done.add(job)
                        self.write_status(phase, job, state='complete', returncode=0, ended_at=self.core.utc_now())
                    except Exception as error:
                        failure = str(error)
                        self.write_status(phase, job, state='failed', returncode=1,
                                          error=failure, ended_at=self.core.utc_now())
                if failure and not active:
                    raise RuntimeError(failure)
        if failure or len(done) != 18:
            raise RuntimeError(failure or 'Incomplete18-job result set')

    def guard_freeze(self):
        self.pre_admission()
        receipt = read(self.control / 'extension_freeze.json')
        frozen = self.core.verify_freeze(self.root, self.batch)
        verify_extension_freeze(receipt, sha(self.resource), sha(self.control / 'validation_freeze.json'))
        if receipt.get('coordinator_recovery_sha256') != self.recovery_binding().get('coordinator_recovery_sha256'):
            raise RuntimeError('Pidfd recovery differs from frozen provenance')
        return frozen

    def run(self):
        with self.recovery.operation_lock(self.control / 'extension_controller'):
            print(json.dumps({'pidfd': PidfdBackend().probe()}), flush=True)
            # Capability probe precedes any new provenance or handoff mutation.
            if self.repair_value is not None:
                immutable_receipt(self.repair_path, self.repair_value)
            self.handoff()
            with self.recovery.operation_lock(self.control / 'controller'):
                immutable_receipt(self.control / 'extension_handoff_complete.json', {'resource_sha256': sha(self.resource)})
                # Rescan after terminating the old controller. Runtime-config
                # discovery also finds a worker forked just before handoff.
                previous = None
                deadline = time.monotonic() + 10
                while True:
                    processes = owned_processes()
                    transitional = any(script_matches(p, self.root, 'run_gencdr_six.py')
                                       and option(p['argv'], '--batch-id') == self.batch for p in processes)
                    live = self.live()
                    sample = sorted((job, p['pid'], p['start_ticks']) for job, rows in live.items() for p in rows)
                    if not transitional and sample == previous:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Worker handoff did not settle; no new workers admitted')
                    previous = sample
                    time.sleep(.5)
                print(json.dumps({'adopted': self.inspect()['live_jobs']}), flush=True)
                if not (self.control / 'validation_freeze.json').exists():
                    self.queue()
                frozen = self.core.freeze_batch(self.root, self.batch)
                immutable_receipt(self.control / 'extension_freeze.json', {
                    'resource_sha256': sha(self.resource),
                    'validation_freeze_sha256': sha(self.control / 'validation_freeze.json'),
                    **self.recovery_binding()})
                self.guard_freeze()
                self.queue(testing=True)
                self.guard_freeze()
                self.core.summarize_batch(self.root, self.batch)
                immutable_receipt(self.control / 'extension_results_provenance.json', {
                    'resource_sha256': sha(self.resource),
                    'extension_freeze_sha256': sha(self.control / 'extension_freeze.json'),
                    'results_json_sha256': sha(self.control / 'results.json'),
                    'results_csv_sha256': sha(self.control / 'results.csv'),
                    **self.recovery_binding()})
                print('EXTENDED_PIPELINE_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('inspect', 'run'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch-id', required=True)
    parser.add_argument('--gpus', nargs='+', required=True)
    parser.add_argument('--max-workers', type=int, default=6)
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('poll-seconds must be1..60')
    coordinator = Coordinator(args.root, args.batch_id, args.gpus, args.max_workers, args.poll_seconds)
    if args.action == 'inspect':
        print(json.dumps(coordinator.inspect(), ensure_ascii=False, indent=2))
    else:
        coordinator.run()


if __name__ == '__main__':
    main()
