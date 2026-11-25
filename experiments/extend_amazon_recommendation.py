"""Adopt live Amazon workers and fill idle GPUs 0..7 without changing their code."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

CODE = Path(__file__).resolve().parents[1]
SIGSTOP = getattr(signal, 'SIGSTOP', 19)
SIGCONT = getattr(signal, 'SIGCONT', 18)


def cli_option(argv, name):
    values = []
    for index, value in enumerate(argv):
        if value == name:
            if index+1 >= len(argv):
                raise RuntimeError(f'Missing process argument: {name}')
            values.append(argv[index+1])
        elif value.startswith(name+'='):
            values.append(value[len(name)+1:])
    if len(values) > 1:
        raise RuntimeError(f'Ambiguous duplicate process argument: {name}')
    return values[0] if values else None


def same_process(a, b):
    return b is not None and all(a.get(k) == b.get(k) for k in ('pid', 'uid', 'start_ticks', 'cwd', 'argv'))


def process_path(process, value, default):
    path = Path(default if value is None else value)
    return (path if path.is_absolute() else Path(process['cwd'])/path).resolve()


def direct_children(pid):
    result = set()
    for thread in (Path('/proc')/str(pid)/'task').iterdir():
        try:
            result.update(map(int, (thread/'children').read_text().split()))
        except FileNotFoundError:
            continue
    return result


def exited_child(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    except FileNotFoundError:
        return True


def settle_workers(discover, children, exited, *, sleep=time.sleep, attempts=200):
    """A forked child must exec into a verified worker or exit before retirement."""
    for _ in range(attempts):
        workers = discover()
        known = {r['process']['pid'] for r in workers}
        transitional = [pid for pid in children() if pid not in known and not exited(pid)]
        if not transitional:
            return workers
        sleep(.05)
    raise RuntimeError('Transitional controller children did not settle; admissions remain closed')


def handoff_controller(process, *, backend, snapshot, discover, persist, close=os.close, sleep=time.sleep):
    """No process groups and no worker signals. Capture children after admissions stop."""
    descriptor = backend.open(process['pid'])
    retired = False
    try:
        if not same_process(process, snapshot(process['pid'])):
            raise RuntimeError('Controller process identity changed before handoff')
        backend.send(descriptor, SIGSTOP)
        try:
            for _ in range(200):
                current = snapshot(process['pid'])
                if not same_process(process, current):
                    raise RuntimeError('Controller identity changed while stopping admissions')
                if current['state'] in ('T', 't'):
                    break
                sleep(.05)
            else:
                raise RuntimeError('Controller did not stop admissions')
            adopted = discover()
            # Receipt is durable before retiring the old scheduler.
            persist(dict(controller=process, adopted=adopted))
        except BaseException:
            if same_process(process, snapshot(process['pid'])):
                backend.send(descriptor, SIGCONT)
            raise
        if not same_process(process, snapshot(process['pid'])):
            raise RuntimeError('Controller identity changed before retirement')
        backend.send(descriptor, signal.SIGTERM)
        try:
            backend.send(descriptor, SIGCONT)  # Deliver pending SIGTERM to a stopped controller.
        except ProcessLookupError:
            pass
        for _ in range(200):
            if not same_process(process, snapshot(process['pid'])):
                retired = True
                break
            sleep(.05)
        if not retired:
            raise RuntimeError('Controller retirement unconfirmed; no new admissions')
        return adopted
    finally:
        close(descriptor)


def job_key(job):
    return job['pair'], job['seed']


def choose_assignment(pending, cards, reserved):
    cards = sorted(cards, key=lambda c: (-c['memory_free_mib'], c['index']))
    return next(((j, c) for j in pending for c in cards if c['index'] in range(8)
        and c['uuid'] not in reserved and not c['pids'] and c['utilization'] <= 5
        and c['memory_free_mib'] >= j.get('minimum_free_mib', 12288)
        and (not j.get('gpu_uuid') or j['gpu_uuid'] == c['uuid'])), None)


def run_queue(group, worker, status, *, adopted, alive, inventory, resources, completed,
              max_workers=8, poll=60, sleep=time.sleep, resource_unavailable=(), writer=None):
    """Already running workers consume slots but never wait for new GPU admission."""
    known = {job_key(j): dict(j) for j in group}
    if len(known) != len(group) or not 1 <= max_workers <= 8:
        raise RuntimeError('Invalid queue identities/cap')
    existing = {}
    for record in adopted:
        identity = job_key(record['job'])
        if identity not in known or identity in existing:
            raise RuntimeError('Unknown or duplicate adopted worker')
        existing[identity] = record
    pending = [j for k, j in known.items() if k not in existing and not completed(j)]
    active, failure = {}, None
    status.mkdir(parents=True, exist_ok=True)
    def record(job, **payload):
        value = dict(job, **payload)
        path = status/f"{job['pair']}_seed{job['seed']}.json"
        if writer:
            writer(path, value)
        else:
            path.write_text(json.dumps(value), encoding='utf-8')
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending or active or existing:
            for identity, prior in list(existing.items()):
                if alive(prior['process']):
                    continue
                existing.pop(identity)
                job = known[identity]
                try:
                    if not completed(job):
                        raise RuntimeError(f'Adopted worker ended without verified completion: {identity}')
                    record(job, state='complete', gpu_uuid=prior['gpu_uuid'], adopted=True)
                except Exception as error:
                    failure = str(error)
                    record(job, state='failed', error=failure, adopted=True)
            for future in [f for f in active if f.done()]:
                job, card = active.pop(future)
                try:
                    future.result()
                    if not completed(job):
                        raise RuntimeError('Worker exited without complete canonical artifacts')
                    record(job, state='complete', gpu_uuid=card['uuid'])
                except Exception as error:
                    if resource_unavailable and isinstance(error, resource_unavailable):
                        pending.append(job)
                        record(job, state='waiting', error=str(error))
                    else:
                        failure = str(error)
                        record(job, state='failed', error=failure)
            if failure:
                if active or existing:
                    if active:
                        wait(active, timeout=poll, return_when=FIRST_COMPLETED)
                    else:
                        sleep(poll)
                    continue
                raise RuntimeError(f'Worker failed; siblings drained: {failure}')
            reason = 'no idle GPU satisfying memory/recorded UUID'
            while pending and len(active)+len(existing) < max_workers:
                reserved = {c['uuid'] for _, c in active.values()} | {r['gpu_uuid'] for r in existing.values()}
                try:
                    cards = inventory()
                    match = choose_assignment(pending, cards, reserved)
                    if match is None:
                        break
                    resources((len(active)+len(existing))*24*2**30)
                except RuntimeError as error:
                    reason = str(error)
                    break
                if any(f.done() for f in active):
                    break
                job, card = match
                pending.remove(job)
                record(job, state='running', gpu_uuid=card['uuid'])
                print(f"START {job['pair']}_seed{job['seed']} GPU={card['index']} {card['uuid']}", flush=True)
                active[pool.submit(worker, job, card)] = job, card
            print(dict(active=len(active), adopted=len(existing), pending=len(pending), maximum_workers=max_workers), flush=True)
            if active:
                wait(active, timeout=poll, return_when=FIRST_COMPLETED)
            elif pending or existing:
                if pending:
                    print(f'WAIT_RESOURCE {reason}; recheck={poll}s', flush=True)
                sleep(poll)


def load_original(tool):
    tool = tool.resolve()
    # This standalone script imports only stdlib before this point, so the
    # frozen original checkout supplies ALL experiment/model dependencies.
    sys.path.insert(0, str(tool))
    spec = importlib.util.spec_from_file_location('_amazon_original', tool/'experiments/run_amazon_recommendation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Extension:
    def __init__(self, args):
        self.args = args
        self.old = load_original(args.original_tool)
        from experiments import extend_gencdr_run as processes
        self.proc = processes
        self.root, self.batch = args.root.resolve(), args.batch_id
        self.folder = self.old.folder_for(self.root, self.batch)
        self.extension = self.folder/'resource_extensions'/args.extension_id
        self.request = self.old.read(self.folder/'request.json')
        self.alignment = Path(self.request['alignment'])
        self.verify_original()

    def verify_original(self):
        old = self.old
        if old.request(self.root, self.batch, self.alignment) != self.request:
            raise RuntimeError('Original source/request changed; use the exact original tool checkout')
        declaration = old.read(self.folder/'declaration.json')
        if declaration['request'] != self.request:
            raise RuntimeError('Original declaration/request binding changed')
        expected = {old.key(j): old.llm_training.config_payload(old.config_for(self.root, self.batch, j)) for j in old.jobs()}
        if declaration['configs'] != expected:
            raise RuntimeError('Frozen training configurations changed')
        for relative, digest in declaration['assets'].items():
            path = (self.root/relative).resolve()
            if not path.is_relative_to(self.root) or old.sha(path) != digest:
                raise RuntimeError('Original asset seal changed')
        for job in old.jobs():
            path = old.job_folder(self.root, self.batch, job)/'runtime.json'
            if old.read(path) != expected[old.key(job)]:
                raise RuntimeError('Original runtime config changed')

    def matches(self, p):
        option = cli_option
        return (p['uid'] == os.getuid() and self.proc.script_matches(p, self.args.original_tool, 'run_amazon_recommendation.py')
            and (option(p['argv'], '--batch-id') or self.old.DEFAULT_BATCH_ID) == self.batch
            and process_path(p, option(p['argv'], '--root'), '/Data/lihb/SIR-CDR') == self.root)

    def controllers(self):
        result = [p for p in self.proc.owned_processes() if self.matches(p)
                and cli_option(p['argv'], '--worker') is None
                and cli_option(p['argv'], '--action') in (None, 'run')]
        for p in result:
            if (Path(p['cwd']).resolve() != self.args.original_tool.resolve()
                    or process_path(p, cli_option(p['argv'], '--alignment'), '') != self.alignment.resolve()):
                raise RuntimeError('Controller working directory/alignment identity mismatch')
        return result

    def workers(self):
        result = []
        known = {job_key(j) for j in self.old.jobs()}
        for p in self.proc.owned_processes():
            if not self.matches(p):
                continue
            phase = cli_option(p['argv'], '--worker')
            if phase is None:
                continue
            if phase not in ('smoke', 'validation', 'test'):
                raise RuntimeError('Unknown worker phase')
            job = dict(pair=cli_option(p['argv'], '--pair'), seed=int(cli_option(p['argv'], '--seed')))
            if (job_key(job) not in known or process_path(p, cli_option(p['argv'], '--alignment'), '') != self.alignment.resolve()
                    or Path(p['cwd']).resolve() != self.args.original_tool.resolve()):
                raise RuntimeError('Worker configuration/root identity mismatch')
            result.append(dict(process=p, job=job, phase=phase, gpu_uuid=self.proc.expected_uuid(p)))
        if len({job_key(r['job']) for r in result}) != len(result):
            raise RuntimeError('Duplicate own workers detected; no extension admission')
        return result

    def resource(self):
        git = self.old.git_identity(CODE)
        if git['status_porcelain']:
            raise RuntimeError('Commit a clean extension tool checkout before production')
        return dict(schema='amazon-eight-gpu-resource-extension-v1', maximum_workers=8,
            allowed_indices=list(range(8)), job_ram_bytes=24*2**30, minimum_free_disk_gib=24,
            original_tool=str(self.args.original_tool.resolve()), root=str(self.root), batch=self.batch,
            request_sha256=self.old.sha(self.folder/'request.json'),
            declaration_sha256=self.old.sha(self.folder/'declaration.json'),
            extension_commit=git['commit'], extension_sha256=self.old.sha(__file__),
            original_source=self.request['source'], models_and_scientific_config_changed=False)

    def inspect(self):
        cards = self.old.query_gpu_inventory()
        return dict(controllers=self.controllers(), workers=self.workers(), effective_resources=self.resource(),
            idle_gpu_indices=[c['index'] for c in cards if choose_assignment([{}], [c], set()) is not None],
            gpu_inventory=cards, free_disk_gib=shutil.disk_usage(self.root).free/2**30)

    def persist_handoff(self, value):
        path = self.extension/'handoff.json'
        if path.exists():
            # A prior attempt may have stopped admissions and then crashed.
            # Preserve its historical snapshot; live workers are rescanned.
            if not same_process(self.old.read(path)['controller'], value['controller']):
                raise RuntimeError('Handoff controller identity differs from original receipt')
            return
        self.old.bind(path, value)

    def run(self):
        old = self.old
        with old.recovery.operation_lock(self.extension/'controller'):
            self.verify_original()
            old.bind(self.extension/'resources.json', self.resource())
            controllers = self.controllers()
            if len(controllers) > 1:
                raise RuntimeError('More than one original controller')
            if controllers:
                backend = self.proc.PidfdBackend()
                backend.probe()
                record = handoff_controller(controllers[0], backend=backend, snapshot=self.proc.process_snapshot,
                    discover=lambda: settle_workers(self.workers, lambda: direct_children(controllers[0]['pid']), exited_child),
                    persist=self.persist_handoff)
                print('CONTROLLER_RETIRED_WORKERS_ADOPTED '+json.dumps(record), flush=True)
            else:
                print('NO_ORIGINAL_CONTROLLER; discovering remaining own workers', flush=True)
            original_wait, original_queue = old.wait_for_workers, old.run_queue
            frozen_request = self.old.sha(self.folder/'request.json')
            frozen_declaration = self.old.sha(self.folder/'declaration.json')

            def verified_workers(root, batch, poll):
                if root != self.root or batch != self.batch or self.controllers():
                    raise RuntimeError('Original controller still active or root/batch changed')
                self.workers()  # Fully validate all live workers; do not drain them.

            def expanded(group, worker, status, **kwargs):
                if (old.sha(self.folder/'request.json') != frozen_request
                        or old.sha(self.folder/'declaration.json') != frozen_declaration):
                    raise RuntimeError('Original declaration changed during extension')
                phase = status.name
                adopted = [r for r in self.workers() if r['phase'] == phase]
                def budget(reserved):
                    if shutil.disk_usage(self.root).free < 24*2**30:
                        raise RuntimeError('WAIT_DISK free space below original 24 GiB gate')
                    old.resources(reserved)
                return run_queue(group, worker, status, adopted=adopted,
                    alive=lambda p: same_process(p, self.proc.process_snapshot(p['pid'])),
                    inventory=old.query_gpu_inventory, resources=budget, completed=kwargs['completed'],
                    max_workers=8, poll=kwargs.get('poll', 60),
                    resource_unavailable=(old.ResourceUnavailable,), writer=old.write_json_atomic)
            try:
                old.wait_for_workers, old.run_queue = verified_workers, expanded
                # The original run acquires its canonical controller lock. Every
                # worker still executes the unmodified original source checkout.
                args = argparse.Namespace(root=self.root, batch_id=self.batch,
                    alignment=self.alignment, poll_seconds=self.args.poll_seconds)
                old.run(args)
            finally:
                old.wait_for_workers, old.run_queue = original_wait, original_queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/Data/lihb/SIR-CDR'))
    parser.add_argument('--original-tool', type=Path, default=Path('/Data/lihb/SIR-CDR-amazon-rec'))
    parser.add_argument('--batch-id', default='amazon-expansion-dual-ndcg-disk24-v1')
    parser.add_argument('--extension-id', default='eight-gpus-v1')
    parser.add_argument('--action', choices=('inspect', 'run'), default='inspect')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5 or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', v)
                                  for v in (args.batch_id, args.extension_id)):
        parser.error('Invalid IDs or poll interval')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    extension = Extension(args)
    if args.action == 'inspect':
        print(json.dumps(extension.inspect(), ensure_ascii=False, indent=2))
    else:
        if sys.platform != 'linux':
            raise RuntimeError('Live adoption requires Linux /proc and pidfd')
        extension.run()


if __name__ == '__main__':
    main()
