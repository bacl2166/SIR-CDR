"""Manually operated six-direction experiment; approved GPU whitelist required."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cdr_framework.experiment_control import (
    SCHEMA, SIX_PAIRS, SEEDS, control_root, validate_batch_id, parse_gpu_ids,
    query_gpu_inventory, admit_gpu_whitelist, assert_gpu_launchable,
    capture_environment, host_resources, git_identity, utc_now, file_sha,
    read_json, write_json_atomic)


def validation_jobs():
    return [{'pair': p, 'seed': s} for s in SEEDS for p in SIX_PAIRS]


def training_memory_requirement(report):
    peak=report.get('peak_cuda_GiB')
    if not isinstance(peak,(int,float)) or not math.isfinite(peak) or peak<=0:
        raise ValueError('Missing real GPU peak memory measurement')
    return math.ceil(peak*1024*1.25+2048)


def worker_commands(root, batch_id, job, phase):
    phases = {'run-validation': ('train', 'validate'), 'run-test': ('test',), 'smoke': ('smoke',)}
    return [[sys.executable, str(Path(root)/'experiments/run_gencdr_shared.py'), '--action', action,
             '--root', str(root), '--pairs', job['pair'], '--seed', str(job['seed']),
             '--batch-id', batch_id, '--device', 'cuda:0'] for action in phases[phase]]


def check_host_pressure():
    value = host_resources()
    if value['memory'].get('MemAvailable', 0) < 16*2**30:
        raise RuntimeError('Host RAM pressure: wait for resources and rerun')
    if value['load'] and value['load'][0] > 1.5*(value['cpu_count'] or 1):
        raise RuntimeError('Host CPU pressure: no new jobs admitted')


def schedule_jobs(jobs, gpu_ids, max_workers, status_dir, worker, *,
                  inventory=query_gpu_inventory, resource_check=check_host_pressure):
    """Bounded queue; never poll a busy external GPU waiting to seize it."""
    gpu_ids = parse_gpu_ids(gpu_ids)
    if not 1 <= max_workers <= len(gpu_ids): raise ValueError('Invalid concurrency')
    pending = list(jobs); active = {}; results = []; in_use = set()
    status_dir = Path(status_dir); status_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending or active:
            blocked = None
            while pending and len(active) < max_workers:
                try:
                    resource_check()
                    cards = inventory()
                    available = admit_gpu_whitelist(gpu_ids, cards)['allowed_gpus']
                    choice = next(((n, g) for n,j in enumerate(pending) for g in available
                                   if g not in in_use and next(c['memory_free_mib'] for c in cards if c['index']==g)>=j.get('minimum_free_mib',4096) and (not j.get('gpu_uuid') or
                                      j['gpu_uuid'] == next(c['uuid'] for c in cards if c['index'] == g))), None)
                    if choice is None: break
                    n, gpu = choice; job = pending.pop(n)
                    assert_gpu_launchable({'requested_gpu_whitelist': gpu_ids, 'gpu_inventory': inventory()}, gpu,
                                          job.get('minimum_free_mib',4096))
                    receipt = {**job, 'gpu': gpu, 'state': 'running', 'started_at': utc_now()}
                    path = status_dir/f"{job['pair']}_seed{job['seed']}.json"
                    write_json_atomic(path, receipt)
                    active[pool.submit(worker, job, gpu)] = (receipt, path)
                    in_use.add(gpu)
                except (RuntimeError, ValueError) as error:
                    blocked = str(error); break
            if not active:
                if pending: raise RuntimeError(blocked or 'Approved cards busy or affinity card unavailable; no job started')
                break
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                receipt, path = active.pop(future); in_use.remove(receipt['gpu'])
                try:
                    detail = future.result() or {}
                    result = {**receipt, 'state': 'complete', 'returncode': 0, 'detail': detail}
                except Exception as error:
                    result = {**receipt, 'state': 'failed', 'returncode': 1, 'error': str(error)}
                result['ended_at'] = utc_now()
                write_json_atomic(path, result); results.append(result)
    return results


def run_validation_queue(queue, **kwargs):
    results = []
    for seed in SEEDS:
        group = queue([j for j in validation_jobs() if j['seed'] == seed], **kwargs)
        results.extend(group)
        if any(r['returncode'] for r in group):
            raise RuntimeError(f'Seed {seed} has failed jobs; later seeds remain closed')
    return results


def child_environment(gpu=None):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', PYTHONIOENCODING='utf-8',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', NVIDIA_TF32_OVERRIDE='0')
    if gpu is not None: env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def run_commands(commands, root, log, gpu=None):
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        for command in commands:
            stream.write(json.dumps({'time': utc_now(), 'command': list(map(str, command)), 'gpu': gpu})+'\n')
            stream.flush()
            process = subprocess.Popen(list(map(str, command)), cwd=root, env=child_environment(gpu),
                                       stdout=stream, stderr=subprocess.STDOUT)
            print(json.dumps({'pid': process.pid, 'gpu': gpu, 'log': str(log)}), flush=True)
            code = process.wait()
            if code: raise RuntimeError(f'Child exit {code}; inspect {log}')
    return {'log': str(log)}


def bind_experiment(root, batch_id, gpus, data_root):
    control = control_root(root, batch_id)
    current = git_identity(root)
    if current['status_porcelain']: raise RuntimeError('Commit/pull a clean Git tree before production work')
    declaration = {'schema': 'six-direction-experiment-v1', 'batch_id': batch_id, 'commit': current['commit'],
                   'pairs': list(SIX_PAIRS), 'seeds': list(SEEDS), 'gpus': list(gpus),
                   'data_root': str(Path(data_root).resolve()), 'paper_alignment_verified': False}
    path = control/'experiment_manifest.json'
    if path.exists() and read_json(path) != declaration:
        raise RuntimeError('Batch declaration changed; use another batch ID')
    if not path.exists(): write_json_atomic(path, declaration)
    return declaration


def asset_commands(root,pair):
    return [[sys.executable,str(Path(root)/'experiments/run_gencdr_shared.py'), '--action',stage,
             '--pairs',pair,'--root',str(root),'--device','cuda:0','--asset-worker']
            for stage in ('embedding','tokenize')]


def pin_asset_models(root,batch_id):
    """Resolve the embedding revision once before any parallel model load."""
    from huggingface_hub import snapshot_download
    from experiments.run_gencdr_shared import paths, asset_identity, model_config, runtime_path
    from experiments.run_llm_cdr import prepare_model
    receipt=control_root(root,batch_id)/'models.json'
    old=read_json(receipt) if receipt.exists() else None
    existing=[]
    for pair in SIX_PAIRS:
        path=paths(root,pair)['control']/'embedding_model.json'
        if path.exists(): existing.append(read_json(path)['revision'])
    if len(set(existing))>1: raise RuntimeError('Existing shared embeddings use different revisions')
    revision=old['embedding_revision'] if old else (existing[0] if existing else 'main')
    snapshot=Path(snapshot_download('Qwen/Qwen3-Embedding-8B',revision=revision))
    if not __import__('re').fullmatch('[0-9a-f]{40}',snapshot.name): raise RuntimeError('Unpinned embedding snapshot')
    value={'embedding_revision':snapshot.name,'embedding_config_sha256':file_sha(snapshot/'config.json'),
           'recommender_revision':model_config(root,SIX_PAIRS[0],42,batch_id).llm_revision}
    if old and old!=value: raise RuntimeError('Pinned model receipt differs')
    if not old: write_json_atomic(receipt,value)
    for pair in SIX_PAIRS:
        p=paths(root,pair)
        expected={'identity':asset_identity(root,pair),'revision':snapshot.name,
                  'model_config_sha256':value['embedding_config_sha256']}
        path=p['control']/'embedding_model.json'
        if path.exists() and read_json(path)!=expected: raise RuntimeError('Existing asset provenance differs')
        if not path.exists(): write_json_atomic(path,expected)
    # Warm the small-model cache once; each pair gets a separately signed runtime later.
    prepare_model(model_config(root,SIX_PAIRS[0],42,batch_id),runtime_path(root,SIX_PAIRS[0],42,batch_id))


def reuse_opposite_embedding(root,pair):
    """Reuse only byte-identical ordered item texts and the pinned encoder recipe."""
    from cdr_framework.experiment_control import can_reuse_embedding
    from experiments.run_gencdr_shared import paths, asset_identity
    import shutil
    opposite=SIX_PAIRS[SIX_PAIRS.index(pair)^1]
    source,target=paths(root,opposite),paths(root,pair)
    if (target['embedding']/'manifest.json').exists(): return True
    if target['embedding'].exists() and any(target['embedding'].iterdir()): return False
    manifest=source['embedding']/'manifest.json'
    if not manifest.exists(): return False
    old=read_json(manifest); model=read_json(target['control']/'embedding_model.json')
    texts=target['processed']/'item_texts.jsonl'
    expected={'input_sha256':file_sha(texts),'model_revision':model['revision'],
              'model_config_sha256':model['model_config_sha256'],'dimension':768,'normalized':True,
              'max_sequence_length':512,'item_count':sum(1 for line in texts.open(encoding='utf-8') if line.strip())}
    if not can_reuse_embedding(old,expected): return False
    if old.get('completed_batches')!=(old['item_count']+old['batch_size']-1)//old['batch_size']:
        return False
    import torch
    tensor=source['embedding']/'text_embeddings.pt'
    matrix=torch.load(tensor,map_location='cpu',weights_only=True)
    if list(matrix.shape)!=old.get('matrix_shape') or not torch.isfinite(matrix).all():
        raise RuntimeError('Reusable embedding matrix is incomplete/nonfinite')
    target['embedding'].mkdir(parents=True,exist_ok=True)
    temporary=target['embedding']/'text_embeddings.pt.part'
    shutil.copyfile(tensor,temporary)
    if file_sha(temporary)!=file_sha(tensor): raise RuntimeError('Embedding copy failed hash check')
    temporary.replace(target['embedding']/'text_embeddings.pt')
    new={**old,**asset_identity(root,pair),'reused_from':opposite,'source_matrix_sha256':file_sha(tensor)}
    write_json_atomic(target['embedding']/'manifest.json',new)
    return True


def execute_phase(root,batch_id,phase,gpus,concurrency,cpu_workers):
    from experiments.run_gencdr_shared import paths, seal, verify_ready, model_config, runtime_path
    from experiments.run_llm_cdr import prepare_model
    control=control_root(root,batch_id)
    logs=root/'logs'/SCHEMA/batch_id
    if phase=='convert':
        def convert(pair):
            return run_commands([[sys.executable,root/'experiments/run_gencdr_shared.py','--action','convert',
                                  '--pairs',pair,'--root',root,'--training-examples','all-prefixes']],
                                root,logs/f'{pair}_convert.log')
        # At most three independent raw readers; memory-heavy default is one.
        results={}
        with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
            futures={pool.submit(convert,p):p for p in SIX_PAIRS}
            for future,pair in futures.items():
                try: results[pair]={'returncode':0,'detail':future.result()}
                except Exception as error: results[pair]={'returncode':1,'error':str(error)}
        write_json_atomic(control/'conversion.json',results)
        if any(r['returncode'] for r in results.values()): raise RuntimeError('Conversion incomplete; inspect logs')
        return
    if phase=='prepare':
        pin_asset_models(root,batch_id)
        def prepare(job,gpu):
            pair=job['pair']; commands=asset_commands(root,pair)
            if (paths(root,pair)['control']/'ready.json').exists():
                verify_ready(root,pair)
            else:
                if reuse_opposite_embedding(root,pair): commands=commands[1:]
                run_commands(commands,root,logs/f'{pair}_prepare.log',gpu)
                seal(root,pair)
            for seed in SEEDS:
                prepare_model(model_config(root,pair,seed,batch_id),runtime_path(root,pair,seed,batch_id))
            return {'ready':True}
        # Finish three forward encodings before the reverse wave can reuse them.
        for pairs in (SIX_PAIRS[::2],SIX_PAIRS[1::2]):
            results=schedule_jobs([{'pair':p,'seed':42,'minimum_free_mib':24576} for p in pairs],gpus,min(concurrency,3),
                                  control/'status/prepare',prepare)
            if any(r['returncode'] for r in results): raise RuntimeError('Asset preparation incomplete')
        return
    from cdr_framework.experiment_control import verify_freeze
    if phase=='run-test':
        frozen=verify_freeze(root,batch_id)
        jobs=[{'pair':j['pair'],'seed':j['seed'],'gpu_uuid':j['gpu_uuid']} for j in frozen['jobs']]
    else: jobs=validation_jobs()
    if phase=='smoke': jobs=[{'pair':p,'seed':42,'minimum_free_mib':24576} for p in SIX_PAIRS]
    if phase=='run-validation':
        for p in SIX_PAIRS:
            smoke=read_json(control/'smoke'/f'{p}.json')
            expected=smoke_identity(root,batch_id,p)
            if smoke['identity']!=expected or not smoke.get('passed'): raise RuntimeError('Missing or stale smoke gate')
    def worker(job,gpu):
        pair,seed=job['pair'],job['seed']
        current=next(c for c in query_gpu_inventory() if c['index']==gpu)
        affinity=control/'affinity'/f'{pair}_seed{seed}.json'
        if phase!='smoke':
            value={'pair':pair,'seed':seed,'gpu_uuid':current['uuid']}
            if affinity.exists() and read_json(affinity)!=value: raise RuntimeError('Job GPU affinity changed')
            if not affinity.exists(): write_json_atomic(affinity,value)
        log=logs/f'{pair}_seed{seed}_{phase}.log'
        result=run_commands(worker_commands(root,batch_id,job,phase),root,log,gpu)
        if phase=='smoke':
            report=None
            for line in log.read_text(encoding='utf-8').splitlines():
                try: value=json.loads(line)
                except ValueError: continue
                if 'peak_cuda_GiB' in value: report=value
            if report is None or report['optimizer_steps']!=0: raise RuntimeError('Smoke report unavailable')
            write_json_atomic(control/'smoke'/f'{pair}.json',
                              {'identity':smoke_identity(root,batch_id,pair),'passed':True,'report':report,
                               'gpu_uuid':current['uuid'],'memory_total_mib':current['memory_total_mib']})
        return result
    def queue(group,**kwargs):
        for job in group:
            if phase in ('run-validation','run-test'):
                measured=read_json(control/'smoke'/f"{job['pair']}.json")
                job['minimum_free_mib']=training_memory_requirement(measured['report'])
            affinity=control/'affinity'/f"{job['pair']}_seed{job['seed']}.json"
            if affinity.exists(): job['gpu_uuid']=read_json(affinity)['gpu_uuid']
        return schedule_jobs(group,gpus,concurrency,control/'status'/phase,worker)
    results=run_validation_queue(queue) if phase=='run-validation' else queue(jobs)
    if any(r['returncode'] for r in results): raise RuntimeError(f'{phase} has failed jobs; inspect status/logs')


def smoke_identity(root,batch_id,pair):
    from experiments.run_gencdr_shared import paths, runtime_path
    return {'commit':git_identity(root)['commit'], 'ready_sha256':file_sha(paths(root,pair)['control']/'ready.json'),
            'runtime_sha256':file_sha(runtime_path(root,pair,42,batch_id))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=('preflight','convert','prepare','smoke',
                        'run-validation','freeze','run-test','status','summarize'))
    parser.add_argument('--batch-id', required=True)
    parser.add_argument('--gpus', nargs='+')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--max-workers', type=int)
    parser.add_argument('--cpu-workers', type=int, default=1)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    validate_batch_id(args.batch_id)
    root = args.root.resolve(); control = control_root(root, args.batch_id)
    from experiments.recover_llm_evaluation import operation_lock
    if args.action in ('status','summarize'):
        if args.action == 'status':
            reports = {str(p.relative_to(control)): read_json(p) for p in control.glob('status/*/*.json')}
            print(json.dumps(reports, ensure_ascii=False, indent=2)); return
        from cdr_framework.experiment_control import summarize_batch
        print(json.dumps(summarize_batch(root, args.batch_id), ensure_ascii=False, indent=2)); return
    with operation_lock(control/'controller'):
        if args.action == 'freeze':
            from cdr_framework.experiment_control import freeze_batch
            print(json.dumps(freeze_batch(root, args.batch_id), ensure_ascii=False)); return
        if not args.gpus or args.data_root is None: parser.error('Execution requires --gpus and --data-root')
        gpus = parse_gpu_ids(args.gpus)
        concurrency = args.max_workers if args.max_workers is not None else len(gpus)
        if not 1 <= concurrency <= len(gpus): parser.error('Invalid max-workers')
        if not 1 <= args.cpu_workers <= 3: parser.error('cpu-workers must be 1..3')
        declaration = bind_experiment(root, args.batch_id, gpus, args.data_root)
        snapshot = capture_environment(root, args.data_root, gpus)
        write_json_atomic(control/'latest_inventory.json', snapshot)
        if not (control/'environment_snapshot.json').exists():
            write_json_atomic(control/'environment_snapshot.json', snapshot)
        if args.action == 'preflight':
            print(json.dumps(snapshot, ensure_ascii=False, indent=2)); return
        if not snapshot['admission']['production_ready']:
            raise RuntimeError('Preflight refused: '+str(snapshot['admission']))
        execute_phase(root, args.batch_id, args.action, gpus, concurrency, args.cpu_workers)


if __name__ == '__main__': main()
