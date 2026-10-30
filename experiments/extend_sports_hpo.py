"""Versioned GPU-pool extension for the immutable Sports/Clothing HPO.

Quiesce only the original controller, drain workers without signalling them,
then transfer the original controller lock. No original identity is rewritten.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OLD_COMMIT = '86440173400ae778a1964814364890a86da3acb3'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load by file, not the experiments package: the training checkout owns that
# package, and the original tool deliberately refuses foreign model imports.
support = load_file('sports_hpo_process_support', ROOT / 'experiments/extend_gencdr_run.py')


def extended_resources(original, extra):
    import re
    if not re.fullmatch(r'GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}', extra):
        raise ValueError('Explicit full physical GPU UUID required')
    if extra in original['gpus']:
        raise ValueError('Requested GPU is already in the original pool')
    return dict(original, gpus=[*original['gpus'], extra])


def option(argv, name, default=None):
    value = support.option(argv, name)
    if value is not None:
        return value
    return next((v.split('=', 1)[1] for v in argv if v.startswith(name + '=')), default)


def matches(process, root, tool, study, action):
    supplied = Path(option(process['argv'], '--root', '/Data/lihb/SIR-CDR'))
    supplied = supplied if supplied.is_absolute() else Path(process['cwd']) / supplied
    return (support.script_matches(process, tool, 'tune_sports_clothing.py')
        and option(process['argv'], '--action') == action
        and option(process['argv'], '--study-id', 'sports-clothing-ndcg-v1') == study
        and supplied.resolve() == Path(root).resolve())


def validate_controller(process, root, tool, study, uid):
    if process['uid'] != uid or not matches(process, root, tool, study, 'run'):
        raise RuntimeError('Original controller owner/script/root/action/study mismatch')


def same_process(current, expected):
    return current is not None and all(current.get(k) == expected.get(k)
                                       for k in ('pid', 'uid', 'start_ticks'))


def send_bound(process, signum, backend=None):
    """pidfd pins the target even if the numeric PID is subsequently recycled."""
    backend = backend or support.PidfdBackend()
    descriptor = backend.open(process['pid'])
    try:
        if not same_process(support.process_snapshot(process['pid']), process):
            raise RuntimeError('Controller exited or PID identity changed; no signal sent')
        backend.send(descriptor, signum)
    finally:
        os.close(descriptor)


def original_tool(root, study):
    # The frozen tool root is not replaced with this extension's checkout.
    import json
    folder = root / 'artifacts/hparam_search' / study
    manifest = json.loads((folder / 'study.json').read_text(encoding='utf-8'))
    tool = Path(manifest['tool']['root']).resolve()
    head = subprocess.check_output(['git', '-C', str(tool), 'rev-parse', 'HEAD'], text=True).strip()
    if head != OLD_COMMIT:
        raise RuntimeError('Only the verified original 8644017 controller is supported')
    old = load_file('frozen_sports_hpo_controller', tool / 'experiments/tune_sports_clothing.py')
    old.verify_study(root, folder)
    return old, tool, folder


def extension_path(folder, extension_id):
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', extension_id):
        raise ValueError('Invalid extension-id')
    path = folder / 'resource_extensions' / extension_id
    if not path.resolve().is_relative_to(folder / 'resource_extensions'):
        raise ValueError('Extension path escapes study')
    return path


def protected_hashes(old, folder):
    files = [folder / name for name in ('study.json', 'study_hash.json', 'resource_policy.json')]
    files += sorted((folder / 'jobs').glob('*/*/*/runtime.yaml'))
    files += sorted((folder / 'jobs').glob('*/*/*/affinity.json'))
    return {str(p.relative_to(folder)): old.sha(p) for p in files}


def verify_protected(old, folder, hashes):
    for relative, expected in hashes.items():
        if old.sha(folder / relative) != expected:
            raise RuntimeError(f'Original frozen file changed: {relative}')


def verify_completed_files(old, folder, rows):
    for row in rows:
        for relative, expected in row.get('files', {}).items():
            path = (folder / relative).resolve()
            if not path.is_relative_to(folder) or old.sha(path) != expected:
                raise RuntimeError(f'Previously completed output changed: {relative}')


def record_completion(old, folder, extension, job, phase, row):
    files = dict(row.get('files', {}))
    if phase == 'validate':
        path = folder / 'jobs' / old.job_key(job) / 'selection.json'
        files[str(path.relative_to(folder))] = old.sha(path)
    old.seal_record(extension / 'completed_jobs' / f'{phase}_{old.job_key(job).replace("/", "_")}.json',
                    {'job': job, 'phase': phase, 'row': row, 'files': files})


def verify_extension(old, root, folder, extension):
    manifest = old.verify_study(root, folder)
    receipt = old.read_record(extension / 'resources.json')
    if (receipt['study_sha256'] != old.sha(folder / 'study.json')
            or receipt['original_resource_sha256'] != old.sha(folder / 'resource_policy.json')
            or receipt['extension_source'] != old.source_identity(ROOT)):
        raise RuntimeError('Extension/study/source identity changed')
    verify_protected(old, folder, receipt['protected_files'])
    verify_completed_files(old, folder, receipt['previously_completed'])
    for path in sorted((extension / 'completed_jobs').glob('*.json')):
        verify_completed_files(old, folder, [old.read_record(path)])
    if (extension / 'drained.json').exists():
        verify_completed_files(old, folder, old.read_record(extension / 'drained.json')['worker_completion'])
    resources = extended_resources(old.read_record(folder / 'resource_policy.json'), receipt['added_gpu'])
    if receipt['effective_resources'] != resources:
        raise RuntimeError('Effective resources differ from the sealed pool extension')
    return manifest, resources


def actors(root, tool, study):
    owned = support.owned_processes()
    controllers = [p for p in owned if matches(p, root, tool, study, 'run')]
    workers = [p for p in owned if matches(p, root, tool, study, 'worker')]
    # Resume only this version's extension workers, never assume their death.
    workers += [p for p in owned if support.script_matches(p, ROOT, 'extend_sports_hpo.py')
        and support.option(p['argv'], '--action') == 'worker'
        and support.option(p['argv'], '--root') == str(root)
        and support.option(p['argv'], '--study-id') == study]
    if len(controllers) > 1:
        raise RuntimeError('Multiple original controllers; no handoff attempted')
    return controllers, workers


def create_extension(old, root, folder, extension, extra):
    if (folder / 'validation_freeze.json').exists() and not (extension / 'resources.json').exists():
        raise RuntimeError('Study already frozen; do not extend or repeat completed HPO')
    if (extension / 'resources.json').exists():
        receipt = old.read_record(extension / 'resources.json')
        if receipt['added_gpu'] != extra:
            raise RuntimeError('Extension GPU changed; use the original invocation')
        verify_extension(old, root, folder, extension)
        return
    old_resources = old.read_record(folder / 'resource_policy.json')
    effective = extended_resources(old_resources, extra)
    original = old.original(root)
    completed = []
    for job in old.all_jobs():
        output = folder / 'jobs' / old.job_key(job) / 'training/manifest.json'
        if output.exists():
            row = old.trained(folder, job, old.config_for(root, folder, job, original), original)
            if row is None:
                raise RuntimeError('Existing training manifest is incomplete')
            completed.append(row)
    old.seal_record(extension / 'resources.json', {
        'schema': 'sports-hpo-resource-extension-v1', 'added_gpu': extra,
        'study_sha256': old.sha(folder / 'study.json'),
        'original_resource_sha256': old.sha(folder / 'resource_policy.json'),
        'extension_source': old.source_identity(ROOT),
        'effective_resources': effective, 'protected_files': protected_hashes(old, folder),
        'previously_completed': completed, 'handoff': 'quiesce-controller; drain-workers; transfer-lock',
        'model_or_objective_changed': False, 'test_action': False})


def completion(old, root, folder, job, phase, modules):
    config = old.config_for(root, folder, job, modules)
    return (old.trained(folder, job, config, modules) if phase == 'train'
            else old.scored(folder, job, config, modules))


def drain_handoff(old, root, tool, folder, extension):
    """Never adopt a running worker: let it finish before changing ownership."""
    modules = old.original(root)
    if (extension / 'retired.json').exists():
        # A crashed successor may have left its own worker. Drain and verify it.
        controllers, _ = actors(root, tool, folder.name)
        if controllers:
            raise RuntimeError('Original controller reappeared after handoff')
        drain_workers(old, root, tool, folder, extension, modules, require_success=False)
        return
    if (extension / 'resume_old.json').exists():
        raise RuntimeError('This extension was cancelled; use a fresh extension-id to retry')
    if not (extension / 'intent.json').exists():
        controllers, workers = actors(root, tool, folder.name)
        controller = controllers[0] if controllers else None
        if controller:
            validate_controller(controller, root, tool, folder.name, os.getuid())
        old.seal_record(extension / 'intent.json', {'controller': controller, 'workers': workers})
    intent = old.read_record(extension / 'intent.json')
    controller = intent['controller']
    if controller and same_process(support.process_snapshot(controller['pid']), controller):
        current = support.process_snapshot(controller['pid'])
        validate_controller(current, root, tool, folder.name, os.getuid())
        if current['state'] != 'T':
            send_bound(controller, signal.SIGSTOP)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            current = support.process_snapshot(controller['pid'])
            if not same_process(current, controller):
                raise RuntimeError('Original controller exited while being quiesced')
            if current['state'] == 'T':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Original controller did not stop; no new task admitted')
    elif controller and not (extension / 'drained.json').exists():
        # Process has gone; workers must still be drained, never presumed done.
        print('Original controller already exited; checking remaining workers', flush=True)
    _, workers = actors(root, tool, folder.name)
    if not (extension / 'quiesced.json').exists():
        old.seal_record(extension / 'quiesced.json', {'controller': controller, 'workers': workers})
    drain_workers(old, root, tool, folder, extension, modules)
    old.seal_record(extension / 'drained.json', {
        'quiesced_sha256': old.sha(extension / 'quiesced.json'),
        'worker_completion': drained_completions(old, root, folder, extension, modules)})
    verify_extension(old, root, folder, extension)
    if controller and same_process(support.process_snapshot(controller['pid']), controller):
        current = support.process_snapshot(controller['pid'])
        if current['state'] != 'T':
            raise RuntimeError('Original controller resumed unexpectedly; refusing takeover')
        old.seal_record(extension / 'retiring.json', {
            'controller': controller, 'drained_sha256': old.sha(extension / 'drained.json')})
        send_bound(controller, signal.SIGTERM)
        try:
            send_bound(controller, signal.SIGCONT)
        except (ProcessLookupError, RuntimeError):
            if same_process(support.process_snapshot(controller['pid']), controller):
                raise
        deadline = time.monotonic() + 10
        while same_process(support.process_snapshot(controller['pid']), controller):
            if time.monotonic() >= deadline:
                raise RuntimeError('Original controller did not exit; successor not admitted')
            time.sleep(.1)
    old.seal_record(extension / 'retired.json', {
        'controller': controller, 'drained_sha256': old.sha(extension / 'drained.json')})


def drain_workers(old, root, tool, folder, extension, modules, require_success=True):
    observed = {}
    for path in sorted((extension / 'worker_observations').glob('*.json')):
        process = old.read_record(path)
        observed[f"{process['pid']}:{process['start_ticks']}"] = process
    # Quiesced snapshot closes the race where a worker exits before first poll.
    def remember(process):
        identity = {key: process[key] for key in ('pid', 'uid', 'start_ticks', 'argv', 'cwd')}
        key = f"{identity['pid']}:{identity['start_ticks']}"
        old.seal_record(extension / 'worker_observations' / f'{key.replace(":", "_")}.json', identity)
        observed[key] = identity
    for process in old.read_record(extension / 'quiesced.json')['workers']:
        remember(process)
    while True:
        _, workers = actors(root, tool, folder.name)
        for process in workers:
            remember(process)
        if not workers:
            break
        print(old.canonical({'DRAINING': [{'pid': p['pid'], 'job': support.option(p['argv'], '--job')}
                                         for p in workers]}), flush=True)
        time.sleep(10)
    for process in observed.values():
        key = support.option(process['argv'], '--job')
        job = old.job_from_key(key)
        phase = support.option(process['argv'], '--phase')
        if job is None or phase not in ('train', 'validate'):
            raise RuntimeError('Unknown worker in drain receipt')
        row = completion(old, root, folder, job, phase, modules)
        if row is None:
            if require_success:
                raise RuntimeError(f'Drained worker did not complete successfully: {key}; no takeover')
            old.seal_record(extension / 'incomplete_workers' / f"{process['pid']}_{process['start_ticks']}.json", {
                'job': job, 'phase': phase, 'recovery': 'resume matching checkpoint under original job lock/affinity'})
        else:
            record_completion(old, folder, extension, job, phase, row)


def drained_completions(old, root, folder, extension, modules):
    observed = [old.read_record(path) for path in sorted((extension / 'worker_observations').glob('*.json'))]
    return [completion(old, root, folder, old.job_from_key(support.option(p['argv'], '--job')),
                       support.option(p['argv'], '--phase'), modules) for p in observed]


def resume_old(old, root, tool, folder, extension):
    if (extension / 'retiring.json').exists() or (extension / 'retired.json').exists():
        raise RuntimeError('Original controller retired; resume successor instead')
    intent = old.read_record(extension / 'intent.json')
    controller = intent['controller']
    if controller is None:
        raise RuntimeError('No original controller to resume')
    current = support.process_snapshot(controller['pid'])
    if not same_process(current, controller):
        raise RuntimeError('Original process identity no longer exists')
    validate_controller(current, root, tool, folder.name, os.getuid())
    old.seal_record(extension / 'resume_old.json', {'controller': controller})
    send_bound(controller, signal.SIGCONT)


def worker(old, root, folder, extension, args):
    if os.environ.get('SIR_HPO_CONTROLLER_PID') != str(os.getppid()):
        raise RuntimeError('Extension worker is internal; use --action run')
    if not (extension / 'takeover.json').exists():
        raise RuntimeError('No completed controller handoff')
    manifest, resources = verify_extension(old, root, folder, extension)
    if args.gpu not in resources['gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu:
        raise RuntimeError('Extension worker has unapproved GPU visibility')
    job = old.job_from_key(args.job)
    if job is None:
        raise RuntimeError('Unknown study job')
    with old.lock(folder / 'locks' / (args.job.replace('/', '_') + '.lock')):
        allowed = old.policy.screen_jobs()
        if (folder / 'promotions.json').exists():
            promotions = old.read_record(folder / 'promotions.json')['promotions']
            allowed = (old.policy.validation_jobs(promotions) if args.phase == 'validate'
                       else allowed + old.policy.confirmation_jobs(promotions))
        if job not in allowed or (args.phase == 'validate' and not (folder / 'promotions.json').exists()):
            raise RuntimeError('Job not admitted by original study stage')
        cards, busy = old.gpu_snapshot()
        old.admit(args.gpu, cards, busy, 0, resources['host_ram_gib'],
                  resources['job_ram_gib'], old.available_ram())
        old.seal(folder / 'jobs' / args.job / 'affinity.json', {'gpu_uuid': args.gpu})
        modules = old.original(root)
        if (folder / 'promotions.json').exists():
            promotions = old.promotions_for(root, folder, modules)
        shared = modules['experiments.run_gencdr_shared']
        shared.verify_gpu_binding(); shared.verify_ready(root, job['pair'])
        config = old.config_for(root, folder, job, modules)
        if old.canonical({'llm_recommendation': modules['cdr_framework.llm_training'].config_payload(config)}) != manifest['configs'][args.job]['payload']:
            raise RuntimeError('Original model configuration changed')
        import torch
        device = torch.device('cuda:0')
        if args.phase == 'train':
            if old.trained(folder, job, config, modules) is None:
                modules['experiments.run_llm_cdr'].smoke(config, device)
                modules['cdr_framework.llm_training'].train(config, device)
                old.flush_tree(config.output_dir)
            if old.trained(folder, job, config, modules) is None:
                raise RuntimeError('Training returned without a verified completion')
        else:
            promotions = old.promotions_for(root, folder, modules)
            if job not in old.policy.validation_jobs(promotions) or old.trained(folder, job, config, modules) is None:
                raise RuntimeError('Only trained finalists may be validated')
            if old.scored(folder, job, config, modules) is not None:
                return
            recovery = modules['experiments.recover_llm_evaluation']
            output = folder / 'jobs' / args.job / 'validation'
            recovery.recover_stage(config, output, 'cache', device)
            recovery.recover_stage(config, output, 'verify', device)
            old.flush_tree(output)
            cache, binding = old.validation_binding(folder, job, config, modules, device)
            old.seal(folder / 'jobs' / args.job / 'selection.json', dict(
                job=job, binding=binding, results=old.score_grid(cache, config, modules),
                proof='full-raw-scores-bitwise-equal; deterministic-CPU-ranker'))


def schedule(old, root, folder, extension, jobs, phase, modules, resources):
    active, pending, failed = [], list(jobs), False
    try:
        while pending or active:
            for entry in list(active):
                process, job, gpu, log = entry
                if process.poll() is None:
                    continue
                log.close(); active.remove(entry)
                error = None
                try:
                    valid = not process.returncode and completion(old, root, folder, job, phase, modules) is not None
                except Exception as exception:
                    valid, error = False, str(exception)
                if not valid:
                    failed = True
                    old.seal(folder / 'failures' / f'{time.time_ns()}.json', {
                        'job': job, 'phase': phase, 'returncode': process.returncode,
                        'extension_id': extension.name, 'error': error})
                else:
                    record_completion(old, folder, extension, job, phase,
                                      completion(old, root, folder, job, phase, modules))
            while pending and not failed and len(active) < resources['max_workers']:
                for job in list(pending):
                    row = completion(old, root, folder, job, phase, modules)
                    if row is not None:
                        record_completion(old, folder, extension, job, phase, row)
                        pending.remove(job)
                if not pending:
                    break
                eligible = old.eligible_pending(folder, pending, active, resources)
                if eligible is None:
                    break
                job, gpu = eligible
                old.seal(folder / 'jobs' / old.job_key(job) / 'affinity.json', {'gpu_uuid': gpu})
                environment = os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES=gpu, SIR_EXPECTED_GPU_UUID=gpu,
                    SIR_HPO_CONTROLLER_PID=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
                log = (folder / 'jobs' / old.job_key(job) / f'{phase}.log').open('ab')
                command = [sys.executable, str(Path(__file__).resolve()), '--action', 'worker',
                    '--root', str(root), '--study-id', folder.name, '--extension-id', extension.name,
                    '--job', old.job_key(job), '--phase', phase, '--gpu', gpu]
                try:
                    process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close(); raise
                active.append((process, job, gpu, log)); pending.remove(job)
                print(old.canonical({'DISPATCHED': old.job_key(job), 'pid': process.pid, 'gpu': gpu}), flush=True)
            if failed and not active:
                raise RuntimeError('Worker failed; other workers drained, no new admissions')
            if pending or active:
                print(old.canonical({'active': len(active), 'pending': len(pending), 'phase': phase}), flush=True)
                time.sleep(5)
    finally:
        for process, _, _, log in active:
            try:
                process.wait()
            finally:
                log.close()


def continue_study(old, root, folder, extension, resources):
    modules = old.original(root)
    schedule(old, root, folder, extension, old.policy.screen_jobs(), 'train', modules, resources)
    rows = [completion(old, root, folder, j, 'train', modules) for j in old.policy.screen_jobs()]
    promotions = old.policy.promote(rows)
    old.seal_record(folder / 'promotions.json', {'promotions': promotions, 'screen_rows': rows})
    confirmations = old.policy.confirmation_jobs(promotions)
    schedule(old, root, folder, extension, confirmations, 'train', modules, resources)
    finalists = old.policy.validation_jobs(promotions)
    schedule(old, root, folder, extension, finalists, 'validate', modules, resources)
    selections = [completion(old, root, folder, j, 'validate', modules) for j in finalists]
    final = old.policy.select_final(selections)
    trained = [completion(old, root, folder, j, 'train', modules) for j in old.policy.screen_jobs() + confirmations]
    if len(trained) != 24 or any(row is None for row in trained):
        raise RuntimeError('Full original 24-run training budget not complete')
    manifest, _ = verify_extension(old, root, folder, extension)
    old.seal(folder / 'validation_freeze.json', {
        'schema': 'sports-clothing-extended-validation-freeze-v1',
        'study_sha256': old.sha(folder / 'study.json'),
        'original_resource_sha256': old.sha(folder / 'resource_policy.json'),
        'extension_resource_sha256': old.sha(extension / 'resources.json'),
        'takeover_sha256': old.sha(extension / 'takeover.json'),
        'promotions_sha256': old.sha(folder / 'promotions.json'), 'all_training': trained,
        'selection_receipts': {old.job_key(j): old.sha(folder / 'jobs' / old.job_key(j) / 'selection.json') for j in finalists},
        'affinity_receipts': {str(p.relative_to(folder)): old.sha(p)
                              for p in sorted((folder / 'jobs').glob('*/*/*/affinity.json'))},
        'failures': {str(p.relative_to(folder)): old.sha(p) for p in sorted((folder / 'failures').glob('*.json'))},
        'validation': final, 'prior_observations': manifest['prior_observations'], 'test_action': False})


def run(old, root, tool, folder, extension, extra):
    with old.lock(folder / 'resource_extension.lock'):
        create_extension(old, root, folder, extension, extra)
        drain_handoff(old, root, tool, folder, extension)
        with old.lock(folder / 'controller.lock'):
            controllers, workers = actors(root, tool, folder.name)
            if controllers or workers:
                raise RuntimeError('Controller/worker still active; no successor admissions')
            _, resources = verify_extension(old, root, folder, extension)
            old.seal_record(extension / 'takeover.json', {
                'resources_sha256': old.sha(extension / 'resources.json'),
                'retired_sha256': old.sha(extension / 'retired.json')})
            continue_study(old, root, folder, extension, resources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=('inspect', 'run', 'worker', 'status', 'resume-old'))
    parser.add_argument('--root', type=Path, default=Path('/Data/lihb/SIR-CDR'))
    parser.add_argument('--study-id', default='sports-clothing-ndcg-v1')
    parser.add_argument('--extension-id', default='gpu7-v1')
    parser.add_argument('--add-gpu')
    parser.add_argument('--job')
    parser.add_argument('--phase', choices=('train', 'validate'))
    parser.add_argument('--gpu')
    args = parser.parse_args()
    root = args.root.resolve()
    old, tool, folder = original_tool(root, args.study_id)
    extension = extension_path(folder, args.extension_id)
    if args.action == 'worker':
        if not args.job or not args.phase:
            parser.error('worker requires --job and --phase')
        worker(old, root, folder, extension, args); return
    if args.action in ('inspect', 'status'):
        import json
        controllers, workers = actors(root, tool, folder.name)
        value = {'controllers': controllers, 'workers': workers,
            'extension_receipts': [p.name for p in sorted(extension.glob('*.json'))],
            'completed_training': len(list((folder / 'jobs').glob('*/*/*/training/manifest.json'))),
            'test_action': False}
        if args.add_gpu:
            resources = extended_resources(old.read_record(folder / 'resource_policy.json'), args.add_gpu)
            value['effective_resources'] = resources
            cards, busy = old.gpu_snapshot()
            try:
                old.admit(args.add_gpu, cards, busy, 0, resources['host_ram_gib'],
                          resources['job_ram_gib'], old.available_ram())
                value['added_gpu_currently_admissible'] = True
            except RuntimeError:
                value['added_gpu_currently_admissible'] = False
            value['pidfd'] = support.PidfdBackend().probe()
        if (extension / 'resources.json').exists():
            verify_extension(old, root, folder, extension)
        print(json.dumps(value, indent=2)); return
    if args.action == 'resume-old':
        with old.lock(folder / 'resource_extension.lock'):
            verify_extension(old, root, folder, extension)
            resume_old(old, root, tool, folder, extension)
        return
    if not args.add_gpu:
        parser.error('run requires the explicitly approved --add-gpu UUID')
    support.PidfdBackend().probe()
    run(old, root, tool, folder, extension, args.add_gpu)


if __name__ == '__main__':
    main()
