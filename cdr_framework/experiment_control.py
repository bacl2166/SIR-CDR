"""Auditable orchestration primitives; importing this module allocates no GPU."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import platform
import getpass
import shutil
import socket
from datetime import datetime, timezone

SCHEMA = 'gencdr-shared-loo-v1'
SIX_PAIRS = ('sports_to_clothing', 'clothing_to_sports', 'phones_to_electronics',
             'electronics_to_phones', 'douban_book_movie', 'douban_movie_book')
SEEDS = (42, 43, 44)


def validate_batch_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', value):
        raise ValueError('Invalid batch ID')
    return value


def control_root(root, batch_id):
    return Path(root)/'artifacts/experiment_control'/SCHEMA/validate_batch_id(batch_id)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json_atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def parse_gpu_ids(values):
    if not values or any(not re.fullmatch(r'\d+', str(v)) for v in values):
        raise ValueError('Supply explicit nonnegative GPU indices')
    result = tuple(map(int, values))
    if len(result) != len(set(result)) or len(result) > 6:
        raise ValueError('Whitelist requires 1..6 unique GPU indices')
    return result


def query_gpu_inventory(run=subprocess.run):
    def query(fields, kind):
        result = run(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                     text=True, capture_output=True, check=True, timeout=20)
        return list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
    try:
        rows = query('index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw', 'gpu')
        cards = []
        for row in rows:
            if len(row) != 9: raise ValueError('Unexpected GPU row')
            numbers = [float(v) for v in row[3:]]
            if not all(math.isfinite(n) and n >= 0 for n in numbers): raise ValueError('Invalid GPU number')
            cards.append(dict(zip(('index', 'uuid', 'name', 'memory_total_mib', 'memory_used_mib',
                                   'memory_free_mib', 'utilization', 'temperature_c', 'power_w'),
                                  [int(row[0]), row[1], row[2], *numbers]), pids=[]))
        if not cards or len({c['uuid'] for c in cards}) != len(cards): raise ValueError('Ambiguous inventory')
        by_uuid = {c['uuid']: c for c in cards}
        for row in query('gpu_uuid,pid,used_gpu_memory', 'compute-apps'):
            if len(row) != 3 or row[0] not in by_uuid: raise ValueError('Ambiguous compute process')
            by_uuid[row[0]]['pids'].append(int(row[1]))
        return cards
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError('Cannot establish GPU occupancy; launch refused') from error


def admit_gpu_whitelist(requested, inventory, minimum_free_mib=4096):
    requested = parse_gpu_ids(requested)
    cards = {c['index']: c for c in inventory}
    if any(i not in cards for i in requested): raise ValueError('Approved GPU absent from inventory')
    allowed, rejected = [], {}
    for i in requested:
        c = cards[i]
        if c['pids'] or c['utilization'] > 5 or c['memory_free_mib'] < minimum_free_mib:
            rejected[str(i)] = 'busy or insufficient free memory'
        else: allowed.append(i)
    return {'allowed_gpus': allowed, 'rejected_gpus': rejected}


def assert_gpu_launchable(snapshot, gpu_id, minimum_free_mib=4096):
    admission = admit_gpu_whitelist(snapshot['requested_gpu_whitelist'], snapshot['gpu_inventory'], minimum_free_mib)
    if gpu_id not in admission['allowed_gpus']:
        raise RuntimeError(f'GPU {gpu_id} is unapproved or busy')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_identity(root, run=subprocess.run):
    def git(*args):
        return run(['git', *args], cwd=root, text=True, capture_output=True, check=True, timeout=20).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'status_porcelain': git('status', '--porcelain'),
            'remote': re.sub(r'(https?://)[^/@]+:[^/@]+@', r'\1[redacted]@', git('remote', 'get-url', 'origin'))}


def host_resources():
    """Linux /proc values in bytes. Unsupported hosts remain diagnostic-only."""
    memory = {}
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, raw = line.split(':', 1)
            memory[key] = int(raw.strip().split()[0])*1024
    return {'cpu_count': os.cpu_count(), 'load': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
            'memory': memory, 'platform': platform.platform()}


def capture_environment(root, personal_data_root, requested_gpus, run=subprocess.run):
    import torch
    root, personal_data_root = Path(root).resolve(), Path(personal_data_root).resolve()
    errors = []
    try: git = git_identity(root, run)
    except (OSError, subprocess.SubprocessError) as error:
        git = {}; errors.append('Git identity unavailable: '+str(error))
    try: cards = query_gpu_inventory(run)
    except RuntimeError as error:
        cards = []; errors.append(str(error))
    admission = admit_gpu_whitelist(requested_gpus, cards) if cards else {'allowed_gpus': [], 'rejected_gpus': {}}
    resources = host_resources()
    filesystems = []
    for path in dict.fromkeys((root, Path.home(), personal_data_root)):
        try:
            usage = shutil.disk_usage(path)
            entry = {'path': str(path), 'free_bytes': usage.free, 'total_bytes': usage.total,
                     'mode': oct(path.stat().st_mode & 0o777), 'writable': os.access(path, os.W_OK)}
            if hasattr(os, 'statvfs'):
                stats = os.statvfs(path); entry['free_inodes'] = stats.f_favail
            filesystems.append(entry)
        except OSError as error: errors.append('Filesystem check: '+str(error))
    cuda = torch.cuda.is_available()
    if not cuda: errors.append('PyTorch CUDA unavailable')
    if git.get('status_porcelain'): errors.append('Git working tree is dirty')
    if resources['memory'].get('MemAvailable', 0) < 16*2**30: errors.append('Need at least 16 GiB available host RAM')
    if not personal_data_root.is_dir() or not os.access(personal_data_root, os.W_OK): errors.append('Personal data directory unavailable')
    if shutil.disk_usage(root).free < 60*2**30: errors.append('Need at least 60 GiB initial free workspace disk')
    admission.update(production_ready=not errors and bool(admission['allowed_gpus']), errors=errors)
    return {'schema': 'six-direction-environment-v1', 'captured_at_utc': utc_now(), 'hostname': socket.gethostname(),
            'user': getpass.getuser(), 'cwd': str(root), 'git': git, 'resources': resources,
            'python': platform.python_version(), 'torch': {'version': torch.__version__, 'cuda_runtime': torch.version.cuda,
                'cuda_available': cuda, 'device_count': torch.cuda.device_count() if cuda else 0},
            'filesystems': filesystems, 'personal_data_root': str(personal_data_root),
            'requested_gpu_whitelist': list(requested_gpus), 'gpu_inventory': cards,
            'admission': admission, 'training_started': False}


def verify_freeze(root, batch_id):
    root = Path(root).resolve()
    value = read_json(control_root(root, batch_id)/'validation_freeze.json')
    expected = {(p, s) for p in SIX_PAIRS for s in SEEDS}
    jobs = value.get('jobs', [])
    if (value.get('schema') != 'six-direction-freeze-v1' or value.get('batch_id') != batch_id
            or value.get('test_released') is not True or len(jobs) != 18
            or {(j['pair'], j['seed']) for j in jobs} != expected):
        raise RuntimeError('Incomplete or invalid 18-job freeze')
    for relative, digest in value['files'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_sha(path) != digest:
            raise RuntimeError(f'Frozen input changed: {relative}')
    return value
