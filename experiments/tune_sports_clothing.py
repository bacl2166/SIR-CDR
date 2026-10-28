"""External, validation-only Sports/Clothing study. Never imports the tool's model code.

The active checkout is read-only except artifacts/hparam_search/<study-id>.
Workers are internal subprocesses; no process is signalled, even after failure.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

TOOL = Path(__file__).resolve().parents[1]
TRAINING_COMMIT = 'a887bd389ff72eb14ec96be0dfb876a61dae9331'
GPU_UUID = r'GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}'
_spec = importlib.util.spec_from_file_location('sports_ndcg_policy', TOOL / 'cdr_framework/ndcg_study.py')
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, default=str, sort_keys=True, allow_nan=False))


def record_digest(payload):
    return hashlib.sha256(json.dumps(canonical(payload), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def seal_record(path, payload):
    """Publish payload and integrity together, so a crash cannot split them."""
    seal(path, {'payload': payload, 'sha256': record_digest(payload)})
    return canonical(payload)


def read_record(path):
    record = read(path)
    if (set(record) != {'payload', 'sha256'}
            or record['sha256'] != record_digest(record['payload'])):
        raise RuntimeError(f'Frozen receipt integrity changed: {path}')
    return record['payload']


def flush_tree(folder):
    """Original engine uses atomic rename; add durability before publishing success."""
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    if os.name != 'nt':
        for path in [p for p in folder.rglob('*') if p.is_dir()] + [folder]:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def seal(path, payload):
    """Durable exclusive publication; never overwrite a receipt or config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.receipt-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(canonical(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read(path) != canonical(payload):
                raise RuntimeError(f'Immutable receipt differs: {path}')
        if os.name != 'nt':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return canonical(payload)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError(f'Already locked: {path}') from error
        try:
            yield
        finally:
            release()


def study_path(root, study_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', study_id):
        raise ValueError('Invalid study-id')
    root = Path(root).resolve()
    result = root / 'artifacts/hparam_search' / study_id
    if not result.resolve().is_relative_to(root / 'artifacts/hparam_search'):
        raise ValueError('Study path escapes artifact root')
    return result


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def source_identity(root):
    if git(root, 'status', '--porcelain'):
        raise RuntimeError(f'Checkout must be clean: {root}')
    return {'root': str(root), 'head': git(root, 'rev-parse', 'HEAD'), 'files': {
        name: sha(root / name) for name in git(root, 'ls-files').splitlines()
        if (root / name).is_file()}}


def original(root):
    """Import originals only, without poisoning cdr_framework with tool imports."""
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    modules = {name: importlib.import_module(name) for name in (
        'experiments.run_gencdr_shared', 'experiments.run_llm_cdr',
        'experiments.recover_llm_evaluation', 'experiments.tune_llm_fusion',
        'experiments.llm_snapshot_scoring', 'cdr_framework.llm_training', 'cdr_framework.experiment_control')}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('cdr_framework', 'experiments') and getattr(module, '__file__', None):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f'Original import escaped active root: {name}')
    return modules


def plan():
    return {'schema': 'sports-clothing-ndcg-study-v1', 'new_training_budget': 24,
            'validation_models': 12, 'screen_jobs': policy.screen_jobs(),
            'candidates': policy.candidate_specs(), 'fusion_settings': policy.fusion_settings(),
            'seeds': list(policy.SEEDS), 'selection': 'three-seed mean N@10/R@10/N@5/R@5',
            'training_commit': TRAINING_COMMIT, 'test_action': False,
            'snapshot_exploration': {
                'selection_records_per_model': 48,
                'ancillary_original_records_per_model': 17,
                'total_records_per_model': 65,
                'unique_settings_per_model': 52,
                'total_records_for_finalists': 780,
                'ancillary_settings_used_for_final_selection': False,
                'excludes_native_epoch_validation_and_prior_exploration': True}}


def job_key(job):
    return f"{job['pair']}/{job['candidate']}/seed{job['seed']}"


def all_jobs():
    return [dict(pair=pair, candidate=c['id'], seed=seed)
            for pair in policy.PAIRS for c in policy.candidate_specs() for seed in policy.SEEDS]


def job_from_key(value):
    return next((j for j in all_jobs() if job_key(j) == value), None)


def config_for(root, folder, job, modules):
    base = modules['experiments.run_gencdr_shared'].model_config(root, job['pair'], job['seed'])
    overrides = next(c['overrides'] for c in policy.candidate_specs() if c['id'] == job['candidate'])
    return replace(base, **(overrides | {'selection_metric': 'NDCG@10',
                                        'output_dir': folder / 'jobs' / job_key(job) / 'training'}))


def asset_files(root, modules):
    shared = modules['experiments.run_gencdr_shared']
    result = {}
    for pair in policy.PAIRS:
        shared.verify_ready(root, pair)
        paths = shared.paths(root, pair)
        for name in ('processed', 'tokenizer', 'embedding'):
            for file in sorted(paths[name].rglob('*')):
                if file.is_file() and not any(part.endswith('.lock') for part in file.parts):
                    result[str(file.resolve())] = sha(file)
        # Control directories may have mutable operation locks/progress reports.
        for name in ('ready.json', 'embedding_model.json', 'tokenizer_execution.json'):
            file = paths['control'] / name
            result[str(file.resolve())] = sha(file)
        # Frozen local Qwen weights/config; no network acquisition.
        model = modules['experiments.run_llm_cdr'].check_cached_model(shared.model_config(root, pair, 42))
        for file in sorted(model.rglob('*')):
            if file.is_file():
                result[str(file.resolve())] = sha(file)
    return result


def prior_bindings(root, prior, modules):
    files, controls = {}, []
    for relative in prior['preserved_diagnostic_reports']:
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            raise ValueError('Prior diagnostic escapes active root')
        files[relative] = sha(path)
    for pair in policy.PAIRS:
        binding = modules['cdr_framework.experiment_control'].collect_validation_binding(root, prior['source_batch'], pair, 42)
        controls.append(binding)
        files.update(binding['files'])
        history = root / binding['output'] / 'history.json'
        files[str(history.relative_to(root)).replace('\\', '/')] = sha(history)
    return {'files': files, 'original_controls': controls}


def initialize(root, folder, prior_path):
    if root == TOOL or root.is_relative_to(TOOL) or TOOL.is_relative_to(root):
        raise RuntimeError('Tool and active training root must be separate checkouts')
    active, tool = source_identity(root), source_identity(TOOL)
    if active['head'] != TRAINING_COMMIT:
        raise RuntimeError('Active training HEAD differs from pinned original')
    modules = original(root)
    prior = read(prior_path)
    if not prior:
        raise RuntimeError('Preserved prior observations are required')
    manifest = plan() | {'source': active, 'tool': tool, 'assets': asset_files(root, modules),
                          'prior_observations': prior, 'prior_sha256': sha(prior_path),
                          'prior_bindings': prior_bindings(root, prior, modules), 'configs': {}}
    training = modules['cdr_framework.llm_training']
    for job in all_jobs():
        payload = {'llm_recommendation': training.config_payload(config_for(root, folder, job, modules))}
        # JSON is a strict YAML subset, accepted by the original from_yaml parser.
        path = folder / 'jobs' / job_key(job) / 'runtime.yaml'
        seal(path, payload)
        manifest['configs'][job_key(job)] = {'path': str(path), 'sha256': sha(path), 'payload': canonical(payload)}
    result = seal(folder / 'study.json', manifest)
    seal(folder / 'study_hash.json', {'sha256': sha(folder / 'study.json')})
    return result


def verify_study(root, folder):
    if read(folder / 'study_hash.json') != {'sha256': sha(folder / 'study.json')}:
        raise RuntimeError('Frozen study manifest changed')
    manifest = read(folder / 'study.json')
    reject_test_artifacts(folder)
    for name in ('resource_policy', 'promotions'):
        if (folder / f'{name}.json').exists():
            read_record(folder / f'{name}.json')
    if manifest.get('source') != source_identity(root) or manifest.get('tool') != source_identity(TOOL):
        raise RuntimeError('Active root or tool source changed')
    if manifest.get('source', {}).get('head') != TRAINING_COMMIT or root == TOOL:
        raise RuntimeError('Invalid original source identity')
    if any(manifest.get(k) != v for k, v in plan().items()):
        raise RuntimeError('Study parameters changed')
    for name, digest in manifest['assets'].items():
        if sha(name) != digest:
            raise RuntimeError(f'Frozen data/model asset changed: {name}')
    for relative, digest in manifest['prior_bindings']['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise RuntimeError('Preserved prior diagnostic/baseline changed')
    for key, entry in manifest['configs'].items():
        expected = folder / 'jobs' / key / 'runtime.yaml'
        if entry['path'] != str(expected) or sha(expected) != entry['sha256'] or read(expected) != entry['payload']:
            raise RuntimeError('Frozen runtime config changed')
    return manifest


def check_training(manifest, state, identity):
    if (manifest.get('complete') is not True or manifest.get('identity') != identity
            or state.get('identity') != identity or manifest.get('best_validation') != state.get('validation')
            or not 0 < state.get('epoch', 0) <= manifest.get('epochs_completed', 0)):
        raise RuntimeError('Training completion/checkpoint identity mismatch')


def reject_test_artifacts(folder):
    for job in all_jobs():
        training = folder / 'jobs' / job_key(job) / 'training'
        if any((training / name).exists() for name in ('test.json', 'test.lock')):
            raise RuntimeError('Existing test result/reservation blocks this validation-only study')


def trained(folder, job, config, modules):
    path = config.output_dir
    if any((path / name).exists() for name in ('test.json', 'test.lock')):
        raise RuntimeError('Existing test result/reservation blocks tuning')
    if not (path / 'manifest.json').exists():
        return None
    manifest = read(path / 'manifest.json')
    if manifest.get('complete') is not True:
        return None
    state, identity = modules['experiments.tune_llm_fusion'].validate_checkpoint(config, path / 'best_model.pt')
    check_training(manifest, state, identity)
    return dict(job, metrics=manifest['best_validation'], files={str(f.relative_to(folder)): sha(f)
                for f in sorted(path.rglob('*')) if f.is_file() and f.name != 'operation.lock'})


def promotions_for(root, folder, modules):
    rows = [trained(folder, j, config_for(root, folder, j, modules), modules) for j in policy.screen_jobs()]
    expected = {'promotions': policy.promote(rows), 'screen_rows': rows}
    if read_record(folder / 'promotions.json') != expected:
        raise RuntimeError('Frozen promotions differ from all sixteen screen artifacts')
    return expected['promotions']


def check_verification(search, stored, verification, identity, hashes):
    if (search.get('identity') != identity or stored.get('identity') != identity
            or search.get('complete') is not True or stored.get('complete') is not True
            or search.get('split') != 'validation'
            or stored.get('snapshot_sha256') != hashes['snapshot_sha256']
            or any(search.get(k) != hashes[k] for k in ('cache_sha256', 'snapshot_sha256'))
            or verification.get('binding') != dict(identity=identity, **hashes)
            or any(verification.get(k) is not True for k in ('raw_scores_equal', 'rankings_equal', 'verified_all_metrics'))
            or not verification.get('metric_differences')
            or any(v != 0 for v in verification['metric_differences'].values())):
        raise RuntimeError('Validation cache/search/verification binding mismatch')


def validation_binding(folder, job, config, modules, device=None):
    import torch
    output = folder / 'jobs' / job_key(job) / 'validation'
    recovery = modules['experiments.recover_llm_evaluation']
    state, identity = modules['experiments.tune_llm_fusion'].validate_checkpoint(config, config.output_dir / 'best_model.pt')
    expected = recovery._identity(config, config.output_dir / 'best_model.pt', identity, torch.device('cpu'))
    identity = read(output / 'search.json')['identity']
    runtime = identity.get('runtime', {})
    affinity = read(folder / 'jobs' / job_key(job) / 'affinity.json')['gpu_uuid']
    if ({k: v for k, v in identity.items() if k != 'runtime'} != {k: v for k, v in expected.items() if k != 'runtime'}
            or runtime.get('device') != 'cuda:0'
            or str(runtime.get('uuid', '')).lower().removeprefix('gpu-') != affinity.lower().removeprefix('gpu-')
            or runtime.get('cuda') != torch.version.cuda):
        raise RuntimeError('Evaluation runtime/checkpoint/source identity changed')
    hashes = {key: sha(output / name) for key, name in (
        ('cache_sha256', 'validation_scores.pt'), ('snapshot_sha256', 'item_states.pt'), ('search_sha256', 'search.json'))}
    stored = torch.load(output / 'validation_scores.pt', map_location='cpu', weights_only=True)
    check_verification(read(output / 'search.json'), stored, read(output / 'verification.json'), identity, hashes)
    if stored['scores']['candidates'].ndim != 2 or len(stored['scores']['positive']) == 0:
        raise RuntimeError('Invalid full-score cache')
    return stored['scores'], dict(identity=identity, **hashes,
        verification_sha256=sha(output / 'verification.json'),
        ranker_sha256=sha(Path(modules['experiments.llm_snapshot_scoring'].__file__)),
        selection_sha256=sha(TOOL / 'cdr_framework/ndcg_study.py'),
        runtime_sha256=sha(folder / 'jobs' / job_key(job) / 'runtime.yaml'))


def score_grid(cache, config, modules):
    import torch
    torch.set_num_threads(1)
    ranking = modules['experiments.llm_snapshot_scoring'].rank_snapshot_parameters
    metric = modules['experiments.tune_llm_fusion'].score_metrics
    return [{'setting': setting, 'metrics': metric(cache, ranking(cache, setting, 20), config.top_ks)}
            for setting in policy.fusion_settings()]


def scored(folder, job, config, modules, device=None):
    path = folder / 'jobs' / job_key(job) / 'selection.json'
    if not path.exists():
        return None
    cache, binding = validation_binding(folder, job, config, modules, device)
    receipt = read(path)
    if receipt.get('binding') != binding or receipt.get('job') != job or receipt.get('proof') != 'full-raw-scores-bitwise-equal; deterministic-CPU-ranker':
        raise RuntimeError('Validation selection receipt changed')
    if [r['setting'] for r in receipt['results']] != policy.fusion_settings():
        raise RuntimeError('Incomplete fusion receipt')
    if receipt['results'] != score_grid(cache, config, modules):
        raise RuntimeError('Fusion receipt differs from deterministic CPU scores')
    return dict(job, results=receipt['results'])


def gpu_snapshot():
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    result = []
    try:
        for line in cards.splitlines():
            uuid, free, utilization = [v.strip() for v in line.split(',')]
            card = dict(uuid=uuid, free_mib=int(free), utilization=float(utilization))
            if not re.fullmatch(GPU_UUID, uuid) or card['free_mib'] < 0 or not 0 <= card['utilization'] <= 100:
                raise ValueError('Invalid GPU measurement')
            result.append(card)
        busy = set(p.strip() for p in processes.splitlines() if p.strip())
        if not result or len({c['uuid'] for c in result}) != len(result) or busy - {c['uuid'] for c in result}:
            raise ValueError('Unknown GPU process identity')
    except ValueError as error:
        raise RuntimeError('Cannot establish physical GPU inventory') from error
    return result, busy


def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    raise RuntimeError('Cannot determine host available RAM')


def admit(uuid, cards, busy, reserved, budget, job_ram, available):
    card = next((c for c in cards if c['uuid'] == uuid), None)
    if (card is None or not isinstance(card.get('utilization'), (int, float))
            or not 0 <= card['utilization'] <= 5 or uuid in busy or card['free_mib'] < 12 * 1024
            or reserved + job_ram > budget or available < reserved + job_ram):
        raise RuntimeError('GPU/RAM admission unavailable')


def orphan_workers(folder):
    if sys.platform != 'linux':
        raise RuntimeError('Live controller requires Linux /proc')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode().split('\0')
            if str(Path(__file__).resolve()) in argv and 'worker' in argv and str(folder.name) in argv:
                raise RuntimeError(f'Active orphan worker {proc.name}; wait for it to exit before resume')
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue


def worker(root, folder, args):
    # Absolutely no original imports / CUDA initialization before physical admission.
    if os.environ.get('SIR_HPO_CONTROLLER_PID') != str(os.getppid()):
        raise RuntimeError('worker is internal; launch through --action run')
    resources = read_record(folder / 'resource_policy.json')
    if args.gpu not in resources['gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu:
        raise RuntimeError('Worker requires controller GPU visibility and approved UUID')
    job = job_from_key(args.job)
    if job is None:
        raise ValueError('Unknown job')
    with lock(folder / 'locks' / (args.job.replace('/', '_') + '.lock')):
        manifest = verify_study(root, folder)
        allowed = policy.screen_jobs()
        if (folder / 'promotions.json').exists():
            promotions = read_record(folder / 'promotions.json')['promotions']
            allowed = policy.validation_jobs(promotions) if args.phase == 'validate' else allowed + policy.confirmation_jobs(promotions)
        if job not in allowed or (args.phase == 'validate' and not (folder / 'promotions.json').exists()):
            raise RuntimeError('Job not admitted by immutable study stage')
        cards, busy = gpu_snapshot()
        admit(args.gpu, cards, busy, 0, resources['host_ram_gib'], resources['job_ram_gib'], available_ram())
        seal(folder / 'jobs' / args.job / 'affinity.json', {'gpu_uuid': args.gpu})
        modules = original(root)
        shared = modules['experiments.run_gencdr_shared']
        if (folder / 'promotions.json').exists():
            promotions = promotions_for(root, folder, modules)
        shared.verify_gpu_binding()
        shared.verify_ready(root, job['pair'])
        config = config_for(root, folder, job, modules)
        if canonical({'llm_recommendation': modules['cdr_framework.llm_training'].config_payload(config)}) != manifest['configs'][args.job]['payload']:
            raise RuntimeError('Original model configuration changed')
        import torch
        device = torch.device('cuda:0')
        if args.phase == 'train':
            if trained(folder, job, config, modules) is None:
                modules['experiments.run_llm_cdr'].smoke(config, device)
                modules['cdr_framework.llm_training'].train(config, device)
                flush_tree(config.output_dir)
            if trained(folder, job, config, modules) is None:
                raise RuntimeError('Training returned without completed manifest')
        else:
            if job not in policy.validation_jobs(promotions) or trained(folder, job, config, modules) is None:
                raise RuntimeError('Validation requires a completed finalist')
            if scored(folder, job, config, modules) is not None:
                return
            recovery = modules['experiments.recover_llm_evaluation']
            output = folder / 'jobs' / args.job / 'validation'
            recovery.recover_stage(config, output, 'cache', device)
            recovery.recover_stage(config, output, 'verify', device)
            flush_tree(output)
            cache, binding = validation_binding(folder, job, config, modules, device)
            results = score_grid(cache, config, modules)
            seal(folder / 'jobs' / args.job / 'selection.json', dict(job=job, binding=binding, results=results,
                 proof='full-raw-scores-bitwise-equal; deterministic-CPU-ranker'))


def resource_policy(args):
    gpus = args.gpus.split(',') if args.gpus else []
    if (not gpus or len(set(gpus)) != len(gpus)
            or any(not re.fullmatch(GPU_UUID, g) for g in gpus)
            or not 1 <= args.max_workers <= 2 or args.max_workers > len(gpus)
            or args.host_ram_gib is None or not math.isfinite(args.host_ram_gib)
            or not math.isfinite(args.job_ram_gib) or args.job_ram_gib <= 0
            or args.host_ram_gib < args.job_ram_gib):
        raise ValueError('Explicit UUID pool, cap<=2 and positive host/job RAM budgets required')
    return dict(gpus=gpus, max_workers=args.max_workers, host_ram_gib=args.host_ram_gib, job_ram_gib=args.job_ram_gib)


def schedule(root, folder, jobs, phase, modules, resources):
    active = []
    try:
        return _schedule(root, folder, jobs, phase, modules, resources, active)
    finally:
        # Admission/identity failures also drain children. Never signal a process.
        for process, _, _, log in active:
            try:
                process.wait()
            finally:
                log.close()


def eligible_pending(folder, pending, active, resources):
    cards, busy = gpu_snapshot()
    for job in pending:
        affinity = folder / 'jobs' / job_key(job) / 'affinity.json'
        candidates = [read(affinity)['gpu_uuid']] if affinity.exists() else resources['gpus']
        for gpu in candidates:
            if gpu not in resources['gpus'] or gpu in [entry[2] for entry in active]:
                continue
            try:
                admit(gpu, cards, busy, len(active) * resources['job_ram_gib'], resources['host_ram_gib'],
                      resources['job_ram_gib'], available_ram())
                return job, gpu
            except RuntimeError:
                continue
    return None


def _schedule(root, folder, jobs, phase, modules, resources, active):
    pending, failed = list(jobs), False
    while pending or active:
        for entry in list(active):
            process, job, gpu, log = entry
            if process.poll() is not None:
                log.close(); active.remove(entry)
                if process.returncode:
                    failed = True
                    seal(folder / 'failures' / f'{time.time_ns()}.json', dict(job=job, phase=phase, returncode=process.returncode))
                else:
                    try:
                        config = config_for(root, folder, job, modules)
                        completed = trained(folder, job, config, modules) if phase == 'train' else scored(folder, job, config, modules)
                        if completed is None:
                            raise RuntimeError('Worker returned without a validated completion')
                    except Exception as error:
                        failed = True
                        seal(folder / 'failures' / f'{time.time_ns()}.json', dict(job=job, phase=phase, error=str(error)))
        while pending and not failed and len(active) < resources['max_workers']:
            job = pending[0]
            config = config_for(root, folder, job, modules)
            done = trained(folder, job, config, modules) if phase == 'train' else scored(folder, job, config, modules)
            if done is not None:
                pending.pop(0); continue
            eligible = eligible_pending(folder, pending, active, resources)
            if eligible is None:
                break
            job, selected = eligible
            affinity = folder / 'jobs' / job_key(job) / 'affinity.json'
            seal(affinity, {'gpu_uuid': selected})
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=selected, SIR_EXPECTED_GPU_UUID=selected,
                               SIR_HPO_CONTROLLER_PID=str(os.getpid()),
                               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               TOKENIZERS_PARALLELISM='false')
            log = (folder / 'jobs' / job_key(job) / f'{phase}.log').open('ab')
            command = [sys.executable, str(Path(__file__).resolve()), '--action', 'worker', '--root', str(root),
                       '--study-id', folder.name, '--job', job_key(job), '--phase', phase, '--gpu', selected]
            try:
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close(); raise
            active.append((process, job, selected, log)); pending.remove(job)
        if failed and not active:
            raise RuntimeError('Worker failed; admissions stopped and siblings finished; inspect failures/logs')
        if pending or active:
            time.sleep(5)


def run(root, folder, args):
    with lock(folder / 'controller.lock'):
        orphan_workers(folder)
        manifest = verify_study(root, folder)
        resources = seal_record(folder / 'resource_policy.json', resource_policy(args))
        modules = original(root)
        schedule(root, folder, policy.screen_jobs(), 'train', modules, resources)
        rows = [trained(folder, j, config_for(root, folder, j, modules), modules) for j in policy.screen_jobs()]
        promotions = policy.promote(rows)
        seal_record(folder / 'promotions.json', {'promotions': promotions, 'screen_rows': rows})
        confirmations = policy.confirmation_jobs(promotions)
        schedule(root, folder, confirmations, 'train', modules, resources)
        validation_jobs = policy.validation_jobs(promotions)
        schedule(root, folder, validation_jobs, 'validate', modules, resources)
        # Parent has no CUDA context: recover identity runtime is bound by worker receipt.
        selections = [scored(folder, j, config_for(root, folder, j, modules), modules) for j in validation_jobs]
        final = policy.select_final(selections)
        trained_rows = [trained(folder, j, config_for(root, folder, j, modules), modules)
                        for j in policy.screen_jobs() + confirmations]
        if len(trained_rows) != 24 or any(r is None for r in trained_rows):
            raise RuntimeError('Full 24-run study completion required')
        verify_study(root, folder)
        seal(folder / 'validation_freeze.json', dict(schema='sports-clothing-validation-freeze-v1',
             study_sha256=sha(folder / 'study.json'), resource_sha256=sha(folder / 'resource_policy.json'),
             promotions_sha256=sha(folder / 'promotions.json'), all_training=trained_rows,
             selection_receipts={job_key(j): sha(folder / 'jobs' / job_key(j) / 'selection.json') for j in validation_jobs},
             failures={str(p.relative_to(folder)): sha(p) for p in sorted((folder / 'failures').glob('*.json'))},
             validation=final, prior_observations=manifest['prior_observations'], test_action=False))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--action', required=True, choices=('plan', 'init', 'status', 'run', 'worker'))
    result.add_argument('--root', type=Path, default=Path('/Data/lihb/SIR-CDR'))
    result.add_argument('--study-id', default='sports-clothing-ndcg-v1')
    result.add_argument('--prior-observations', type=Path)
    result.add_argument('--gpus')
    result.add_argument('--max-workers', type=int, default=2)
    result.add_argument('--host-ram-gib', type=float)
    result.add_argument('--job-ram-gib', type=float, default=24.)
    result.add_argument('--job')
    result.add_argument('--phase', choices=('train', 'validate'))
    result.add_argument('--gpu')
    return result


def main():
    args = parser().parse_args()
    root = args.root.resolve()
    folder = study_path(root, args.study_id)
    if args.action == 'plan':
        value = plan()
    elif args.action == 'init':
        if not args.prior_observations:
            raise ValueError('init requires --prior-observations')
        with lock(folder / 'controller.lock'):
            value = initialize(root, folder, args.prior_observations)
    elif args.action == 'status':
        verify_study(root, folder)
        value = dict(frozen=(folder / 'validation_freeze.json').exists(),
                     manifests=len(list((folder / 'jobs').glob('*/*/*/training/manifest.json'))),
                     selections=len(list((folder / 'jobs').glob('*/*/*/selection.json'))),
                     failures=[read(p) for p in sorted((folder / 'failures').glob('*.json'))])
    elif args.action == 'worker':
        worker(root, folder, args); return
    else:
        run(root, folder, args); value = read(folder / 'validation_freeze.json')
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
