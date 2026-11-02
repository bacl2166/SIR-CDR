"""Continue the frozen study after an audited handoff from the GPU7 successor.

Only resource allocation changes. Original Python files, study/configuration
receipts, affinities, completed models, promotion and fusion policy are untouched.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
import importlib.util
_spec = importlib.util.spec_from_file_location('idle_pool_base', ROOT / 'experiments/extend_sports_hpo.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
PREVIOUS_COMMIT = '47bdc1731f89d074cbd4fdfc0bb76399e4dc9929'


def resources_for(previous, additions, cap, host_ram):
    if (not additions or len(set(additions)) != len(additions)
            or any(not re.fullmatch(r'GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}', g)
                   or g in previous['gpus'] for g in additions)
            or type(cap) is not int or not 1 <= cap <= min(6, len(previous['gpus']) + len(additions))
            or not math.isfinite(host_ram) or not cap * previous['job_ram_gib'] <= host_ram <= 192.):
        raise ValueError('Require new full UUIDs, cap<=6 and sufficient host RAM capped at192 GiB')
    return dict(previous, gpus=[*previous['gpus'], *additions],
                max_workers=cap, host_ram_gib=float(host_ram))


def matches(process, root, tool, script, study, action, identifier):
    argv = process['argv']
    supplied = Path(base.option(argv, '--root', '/Data/lihb/SIR-CDR'))
    supplied = supplied if supplied.is_absolute() else Path(process['cwd']) / supplied
    id_option, default = (('--extension-id', 'gpu7-v1') if script == 'extend_sports_hpo.py'
                          else ('--expansion-id', 'idle-gpus-v2'))
    return (base.support.script_matches(process, tool, script)
        and base.option(argv, '--action') == action
        and base.option(argv, '--study-id', 'sports-clothing-ndcg-v1') == study
        and base.option(argv, id_option, default) == identifier
        and supplied.resolve() == root.resolve())


def validate_predecessor(process, root, tool, study, identifier, uid):
    if process['uid'] != uid or not matches(process, root, tool,
            'extend_sports_hpo.py', study, 'run', identifier):
        raise RuntimeError('Predecessor controller identity mismatch; no signal sent')


class Expansion:
    def __init__(self, args):
        self.root = args.root.resolve()
        self.old, self.original_root, self.folder = base.original_tool(self.root, args.study_id)
        self.path = base.extension_path(self.folder, args.expansion_id)
        self.previous_root = args.previous_root.resolve()
        self.previous_id = args.previous_id
        self.previous_path = base.extension_path(self.folder, self.previous_id)
        if self.path == self.previous_path or self.previous_root == ROOT:
            raise RuntimeError('Predecessor and successor must use distinct directories/receipts')
        if subprocess.check_output(['git', '-C', str(self.previous_root), 'rev-parse', 'HEAD'],
                                   text=True).strip() != PREVIOUS_COMMIT:
            raise RuntimeError('Predecessor must remain at the published 47bdc17 commit')
        # Check clean source identity before loading any predecessor Python code.
        identity = self.old.source_identity(self.previous_root)
        receipt = self.old.read_record(self.previous_path / 'resources.json')
        if receipt['extension_source'] != identity:
            raise RuntimeError('Predecessor source differs from its frozen receipt')
        self.previous = base.load_file('verified_gpu7_predecessor',
            self.previous_root / 'experiments/extend_sports_hpo.py')
        self.previous.verify_extension(self.old, self.root, self.folder, self.previous_path)
        if not (self.previous_path / 'takeover.json').is_file():
            raise RuntimeError('GPU7 predecessor has not completed its original handoff')

    def previous_resources(self):
        return self.previous.verify_extension(self.old, self.root, self.folder, self.previous_path)[1]

    def actors(self, allow_successor=False):
        controllers, workers = [], []
        for process in base.support.owned_processes():
            if process['pid'] == os.getpid():
                continue
            recognized = False
            if base.matches(process, self.root, self.original_root, self.folder.name, 'run'):
                raise RuntimeError('Original retired controller reappeared; no handoff')
            for tool, script, identifier in (
                    (self.previous_root, 'extend_sports_hpo.py', self.previous_id),
                    (ROOT, 'expand_sports_hpo.py', self.path.name)):
                if matches(process, self.root, tool, script, self.folder.name, 'run', identifier):
                    recognized = True
                    if script != 'extend_sports_hpo.py' and not allow_successor:
                        raise RuntimeError('Another expansion controller is already running')
                    controllers.append(process)
                if matches(process, self.root, tool, script, self.folder.name, 'worker', identifier):
                    recognized = True
                    workers.append(process)
            if base.matches(process, self.root, self.original_root, self.folder.name, 'worker'):
                recognized = True
                workers.append(process)
            if not recognized:
                script = next((Path(value) for value in process['argv'][1:] if value.endswith('.py')), None)
                supplied = Path(base.option(process['argv'], '--root', '/Data/lihb/SIR-CDR'))
                supplied = supplied if supplied.is_absolute() else Path(process['cwd']) / supplied
                if (script and script.name in ('expand_sports_hpo.py', 'extend_sports_hpo.py',
                                               'tune_sports_clothing.py')
                        and supplied.resolve() == self.root.resolve()
                        and base.option(process['argv'], '--study-id', 'sports-clothing-ndcg-v1') == self.folder.name
                        and base.option(process['argv'], '--action') in ('run', 'worker')):
                    raise RuntimeError('Unknown same-study controller/worker; preserve original invocation and drain first')
        if len(controllers) > 1:
            raise RuntimeError('Multiple predecessor controllers')
        return controllers, workers

    def validate_target(self, process):
        validate_predecessor(process, self.root, self.previous_root,
            self.folder.name, self.previous_id, os.getuid())

    def create(self, additions, cap, host_ram):
        old = self.old
        if (self.folder / 'validation_freeze.json').exists():
            raise RuntimeError('Study already frozen; no further HPO admissions')
        resources = resources_for(self.previous_resources(), additions, cap, host_ram)
        if (self.path / 'resources.json').exists():
            if self.verify()[1] != resources:
                raise RuntimeError('Expansion invocation differs from sealed resources')
            return
        modules = old.original(self.root)
        completed = []
        for job in old.all_jobs():
            if (self.folder / 'jobs' / old.job_key(job) / 'training/manifest.json').exists():
                row = self.completion(job, 'train', modules)
                if row is None:
                    raise RuntimeError('Incomplete existing training manifest')
                completed.append(row)
        old.seal_record(self.path / 'resources.json', {
            'schema': 'sports-hpo-idle-pool-v2', 'source': old.source_identity(ROOT),
            'predecessor_root': str(self.previous_root), 'predecessor_id': self.previous_id,
            'predecessor_files': {name: old.sha(self.previous_path / name)
                                  for name in ('resources.json', 'takeover.json')},
            'study_sha256': old.sha(self.folder / 'study.json'),
            'protected_files': base.protected_hashes(old, self.folder),
            'previously_completed': completed, 'effective_resources': resources,
            'additions': additions, 'model_or_objective_changed': False, 'test_action': False})

    def verify(self):
        old = self.old
        manifest = old.verify_study(self.root, self.folder)
        previous = self.previous_resources()
        receipt = old.read_record(self.path / 'resources.json')
        if (receipt['source'] != old.source_identity(ROOT)
                or receipt['study_sha256'] != old.sha(self.folder / 'study.json')
                or receipt['predecessor_root'] != str(self.previous_root)
                or receipt['predecessor_id'] != self.previous_id
                or any(old.sha(self.previous_path / name) != digest
                       for name, digest in receipt['predecessor_files'].items())):
            raise RuntimeError('Expansion source/predecessor/study binding changed')
        base.verify_protected(old, self.folder, receipt['protected_files'])
        base.verify_completed_files(old, self.folder, receipt['previously_completed'])
        for path in sorted((self.path / 'completed_jobs').glob('*.json')):
            base.verify_completed_files(old, self.folder, [old.read_record(path)])
        resources = receipt['effective_resources']
        if resources != resources_for(previous, receipt['additions'], resources['max_workers'],
                                       resources['host_ram_gib']):
            raise RuntimeError('Effective resource policy changed')
        return manifest, resources

    def completion(self, job, phase, modules):
        return base.completion(self.old, self.root, self.folder, job, phase, modules)

    def record(self, job, phase, row):
        base.record_completion(self.old, self.folder, self.path, job, phase, row)

    def remember_worker(self, process):
        identity = {key: process[key] for key in ('pid', 'uid', 'start_ticks', 'cwd', 'argv')}
        self.old.seal_record(self.path / 'observed_workers' /
            f"{identity['pid']}_{identity['start_ticks']}.json", identity)

    def drain(self):
        old = self.old
        modules = old.original(self.root)
        while True:
            _, workers = self.actors()
            for worker in workers:
                self.remember_worker(worker)
            if not workers:
                break
            print(old.canonical({'DRAINING': [dict(pid=p['pid'],
                job=base.option(p['argv'], '--job')) for p in workers]}), flush=True)
            time.sleep(10)
        for path in sorted((self.path / 'observed_workers').glob('*.json')):
            process = old.read_record(path)
            job = old.job_from_key(base.option(process['argv'], '--job'))
            phase = base.option(process['argv'], '--phase')
            if job is None or phase not in ('train', 'validate'):
                raise RuntimeError('Unknown drained worker job/stage')
            row = self.completion(job, phase, modules)
            if row is None:
                old.seal_record(self.path / 'incomplete_workers' / path.name,
                    {'job': job, 'phase': phase, 'recovery': 'original checkpoint/affinity/job lock'})
            else:
                self.record(job, phase, row)

    def handoff(self):
        old = self.old
        if not (self.path / 'intent.json').exists():
            controllers, workers = self.actors()
            controller = controllers[0] if controllers else None
            if controller:
                self.validate_target(controller)
            old.seal_record(self.path / 'intent.json', {'controller': controller})
            for process in workers:
                self.remember_worker(process)
        target = old.read_record(self.path / 'intent.json')['controller']
        current = base.support.process_snapshot(target['pid']) if target else None
        if target and base.same_process(current, target):
            self.validate_target(current)
            if (self.path / 'retired.json').exists():
                raise RuntimeError('Retired predecessor is still alive')
            if current['state'] != 'T':
                base.send_bound(target, signal.SIGSTOP)
            deadline = time.monotonic() + 10
            while True:
                current = base.support.process_snapshot(target['pid'])
                if not base.same_process(current, target):
                    break
                if current['state'] == 'T':
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Controller failed to quiesce')
                time.sleep(.1)
        self.drain()
        controllers, workers = self.actors()
        if workers or any(not base.same_process(p, target) for p in controllers):
            raise RuntimeError('Unexpected controller/worker appeared during handoff')
        current = base.support.process_snapshot(target['pid']) if target else None
        if target and base.same_process(current, target):
            self.validate_target(current)
            if current['state'] != 'T':
                raise RuntimeError('Predecessor resumed unexpectedly')
            old.seal_record(self.path / 'retiring.json', {'controller': target})
            base.send_bound(target, signal.SIGTERM)
            try:
                base.send_bound(target, signal.SIGCONT)
            except (ProcessLookupError, RuntimeError):
                if base.same_process(base.support.process_snapshot(target['pid']), target):
                    raise
            deadline = time.monotonic() + 10
            while base.same_process(base.support.process_snapshot(target['pid']), target):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Predecessor did not exit; no successor admission')
                time.sleep(.1)
        old.seal_record(self.path / 'retired.json', {'controller': target})

    def worker(self, args):
        old = self.old
        if os.environ.get('SIR_HPO_CONTROLLER_PID') != str(os.getppid()):
            raise RuntimeError('Internal worker; use --action run')
        if not (self.path / 'takeover.json').exists():
            raise RuntimeError('Controller handoff has not completed')
        manifest, resources = self.verify()
        if args.gpu not in resources['gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu:
            raise RuntimeError('Unapproved worker GPU visibility')
        job = old.job_from_key(args.job)
        if job is None:
            raise RuntimeError('Unknown study job')
        with old.lock(self.folder / 'locks' / (args.job.replace('/', '_') + '.lock')):
            allowed = old.policy.screen_jobs()
            if (self.folder / 'promotions.json').exists():
                promotions = old.read_record(self.folder / 'promotions.json')['promotions']
                allowed = (old.policy.validation_jobs(promotions) if args.phase == 'validate'
                           else allowed + old.policy.confirmation_jobs(promotions))
            if job not in allowed or (args.phase == 'validate' and not (self.folder / 'promotions.json').exists()):
                raise RuntimeError('Job not admitted by original study stage')
            cards, busy = old.gpu_snapshot()
            old.admit(args.gpu, cards, busy, 0, resources['host_ram_gib'],
                      resources['job_ram_gib'], old.available_ram())
            old.seal(self.folder / 'jobs' / args.job / 'affinity.json', {'gpu_uuid': args.gpu})
            modules = old.original(self.root)
            if (self.folder / 'promotions.json').exists():
                old.promotions_for(self.root, self.folder, modules)
            shared = modules['experiments.run_gencdr_shared']
            shared.verify_gpu_binding(); shared.verify_ready(self.root, job['pair'])
            config = old.config_for(self.root, self.folder, job, modules)
            if old.canonical({'llm_recommendation': modules['cdr_framework.llm_training'].config_payload(config)}) != manifest['configs'][args.job]['payload']:
                raise RuntimeError('Original training configuration changed')
            import torch
            device = torch.device('cuda:0')
            if args.phase == 'train':
                if self.completion(job, 'train', modules) is None:
                    modules['experiments.run_llm_cdr'].smoke(config, device)
                    modules['cdr_framework.llm_training'].train(config, device)
                    old.flush_tree(config.output_dir)
                if self.completion(job, 'train', modules) is None:
                    raise RuntimeError('Training returned without verified completion')
            else:
                promotions = old.promotions_for(self.root, self.folder, modules)
                if job not in old.policy.validation_jobs(promotions) or self.completion(job, 'train', modules) is None:
                    raise RuntimeError('Only trained finalists can be scored')
                if self.completion(job, 'validate', modules) is not None:
                    return
                output = self.folder / 'jobs' / args.job / 'validation'
                recovery = modules['experiments.recover_llm_evaluation']
                recovery.recover_stage(config, output, 'cache', device)
                recovery.recover_stage(config, output, 'verify', device)
                old.flush_tree(output)
                cache, binding = old.validation_binding(self.folder, job, config, modules, device)
                old.seal(self.folder / 'jobs' / args.job / 'selection.json', dict(
                    job=job, binding=binding, results=old.score_grid(cache, config, modules),
                    proof='full-raw-scores-bitwise-equal; deterministic-CPU-ranker'))

    def schedule(self, jobs, phase, modules, resources):
        old = self.old
        pending, active, failed = list(jobs), [], False
        try:
            while pending or active:
                for entry in list(active):
                    process, job, gpu, log = entry
                    if process.poll() is None:
                        continue
                    log.close(); active.remove(entry)
                    error, row = None, None
                    try:
                        if not process.returncode:
                            row = self.completion(job, phase, modules)
                    except Exception as exception:
                        error = str(exception)
                    if row is None:
                        failed = True
                        old.seal(self.folder / 'failures' / f'{time.time_ns()}.json', dict(
                            job=job, phase=phase, returncode=process.returncode,
                            expansion_id=self.path.name, error=error))
                    else:
                        self.record(job, phase, row)
                while pending and not failed and len(active) < resources['max_workers']:
                    for job in list(pending):
                        row = self.completion(job, phase, modules)
                        if row is not None:
                            self.record(job, phase, row); pending.remove(job)
                    if not pending:
                        break
                    eligible = old.eligible_pending(self.folder, pending, active, resources)
                    if eligible is None:
                        break
                    job, gpu = eligible
                    old.seal(self.folder / 'jobs' / old.job_key(job) / 'affinity.json', {'gpu_uuid': gpu})
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=gpu, SIR_EXPECTED_GPU_UUID=gpu,
                        SIR_HPO_CONTROLLER_PID=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        TOKENIZERS_PARALLELISM='false')
                    log = (self.folder / 'jobs' / old.job_key(job) / f'{phase}.log').open('ab')
                    command = [sys.executable, str(Path(__file__).resolve()), '--action', 'worker',
                        '--root', str(self.root), '--study-id', self.folder.name,
                        '--expansion-id', self.path.name, '--previous-root', str(self.previous_root),
                        '--previous-id', self.previous_id, '--job', old.job_key(job), '--phase', phase, '--gpu', gpu]
                    try:
                        process = subprocess.Popen(command, cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT)
                    except BaseException:
                        log.close(); raise
                    active.append((process, job, gpu, log)); pending.remove(job)
                    print(old.canonical({'DISPATCHED': old.job_key(job), 'pid': process.pid, 'gpu': gpu}), flush=True)
                if failed and not active:
                    raise RuntimeError('Worker failed; siblings drained, new admissions stopped')
                if pending or active:
                    print(old.canonical({'active': len(active), 'pending': len(pending), 'phase': phase}), flush=True)
                    time.sleep(5)
        finally:
            for process, _, _, log in active:
                try:
                    process.wait()
                finally:
                    log.close()

    def continue_study(self, resources):
        old = self.old
        modules = old.original(self.root)
        screens = old.policy.screen_jobs()
        self.schedule(screens, 'train', modules, resources)
        rows = [self.completion(job, 'train', modules) for job in screens]
        promotions = old.policy.promote(rows)
        old.seal_record(self.folder / 'promotions.json', {'promotions': promotions, 'screen_rows': rows})
        confirmations = old.policy.confirmation_jobs(promotions)
        self.schedule(confirmations, 'train', modules, resources)
        finalists = old.policy.validation_jobs(promotions)
        self.schedule(finalists, 'validate', modules, resources)
        selected = [self.completion(job, 'validate', modules) for job in finalists]
        final = old.policy.select_final(selected)
        trained = [self.completion(job, 'train', modules) for job in screens + confirmations]
        if len(trained) != 24 or any(row is None for row in trained):
            raise RuntimeError('Original full training budget is not complete')
        manifest, _ = self.verify()
        old.seal(self.folder / 'validation_freeze.json', dict(
            schema='sports-clothing-idle-pool-freeze-v2',
            study_sha256=old.sha(self.folder / 'study.json'),
            original_resource_sha256=old.sha(self.folder / 'resource_policy.json'),
            expansion_resource_sha256=old.sha(self.path / 'resources.json'),
            takeover_sha256=old.sha(self.path / 'takeover.json'),
            promotions_sha256=old.sha(self.folder / 'promotions.json'),
            all_training=trained, selection_receipts={old.job_key(j): old.sha(
                self.folder / 'jobs' / old.job_key(j) / 'selection.json') for j in finalists},
            affinity_receipts={str(p.relative_to(self.folder)): old.sha(p)
                for p in sorted((self.folder / 'jobs').glob('*/*/*/affinity.json'))},
            failures={str(p.relative_to(self.folder)): old.sha(p)
                for p in sorted((self.folder / 'failures').glob('*.json'))},
            validation=final, prior_observations=manifest['prior_observations'], test_action=False))

    def run(self, args):
        old = self.old
        # Predecessor holds resource_extension.lock until retired. A different
        # gate serializes handoff attempts before acquiring the original locks.
        with old.lock(self.folder / 'resource_expansion.lock'):
            self.create(args.add_gpus, args.max_workers, args.host_ram_gib)
            self.handoff()
            with old.lock(self.folder / 'resource_extension.lock'), old.lock(self.folder / 'controller.lock'):
                controllers, workers = self.actors()
                if controllers or workers:
                    raise RuntimeError('Controller/worker remains; no successor admission')
                _, resources = self.verify()
                old.seal_record(self.path / 'takeover.json', dict(
                    resources_sha256=old.sha(self.path / 'resources.json'),
                    retired_sha256=old.sha(self.path / 'retired.json')))
                self.continue_study(resources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=('inspect', 'run', 'worker', 'status'))
    parser.add_argument('--root', type=Path, default=Path('/Data/lihb/SIR-CDR'))
    parser.add_argument('--study-id', default='sports-clothing-ndcg-v1')
    parser.add_argument('--expansion-id', default='idle-gpus-v2')
    parser.add_argument('--previous-root', type=Path, default=Path('/Data/lihb/SIR-CDR-hpo-gpu7'))
    parser.add_argument('--previous-id', default='gpu7-v1')
    parser.add_argument('--add-gpus', nargs='+')
    parser.add_argument('--max-workers', type=int, default=6)
    parser.add_argument('--host-ram-gib', type=float, default=192.)
    parser.add_argument('--job')
    parser.add_argument('--phase', choices=('train', 'validate'))
    parser.add_argument('--gpu')
    args = parser.parse_args()
    expansion = Expansion(args)
    if args.action == 'worker':
        if not args.job or not args.phase or not args.gpu:
            parser.error('worker requires job, phase and GPU')
        expansion.worker(args); return
    if args.action in ('inspect', 'status'):
        controllers, workers = expansion.actors(allow_successor=True)
        if (expansion.path / 'resources.json').exists():
            _, resources = expansion.verify()
        elif args.add_gpus:
            resources = resources_for(expansion.previous_resources(), args.add_gpus,
                                      args.max_workers, args.host_ram_gib)
        else:
            resources = expansion.previous_resources()
        cards, busy = expansion.old.gpu_snapshot()
        eligible = []
        for gpu in resources['gpus']:
            try:
                expansion.old.admit(gpu, cards, busy, 0, resources['host_ram_gib'],
                                    resources['job_ram_gib'], expansion.old.available_ram())
                eligible.append(gpu)
            except RuntimeError:
                pass
        print(json.dumps(dict(controllers=controllers, workers=workers,
            effective_resources=resources, currently_admissible=eligible,
            completed_training=len(list((expansion.folder / 'jobs').glob('*/*/*/training/manifest.json'))),
            handoff_receipts=[p.name for p in sorted(expansion.path.glob('*.json'))],
            test_action=False), indent=2)); return
    if not args.add_gpus:
        parser.error('run requires explicitly approved --add-gpus UUIDs')
    base.support.PidfdBackend().probe()
    expansion.run(args)


if __name__ == '__main__':
    main()
