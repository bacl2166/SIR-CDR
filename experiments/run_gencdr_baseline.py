"""Wait for SIR-CDR completion, then run a separately labelled GenCDR reconstruction."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdr_framework.baselines.gencdr import MODEL_LABEL, UPSTREAM_REVISION
from cdr_framework.baselines.gencdr.assets import ExistingAssets, read
from cdr_framework.baselines.gencdr.protocol import (
    PAIRS, SEEDS, REVERSE, bind, jobs, key, sir_completion, summarize, verify_files,
)
from cdr_framework.baselines.gencdr.training import BaselineConfig, train_job, evaluate_job
from cdr_framework.datasets.gencdr_protocol import sha
from cdr_framework.experiment_control import (
    git_identity, query_gpu_inventory, validate_batch_id, write_json_atomic, host_resources,
)
from experiments.recover_llm_evaluation import operation_lock

SOURCE_BATCH = 'amazon-expansion-dual-ndcg-disk24-v1'
BASELINE_BATCH = 'gencdr-reconstruction-amazon-v1'
MINIMUM_FREE_MIB = 24576
MINIMUM_DISK_GIB = 24


class ResourceWait(RuntimeError):
    pass


def eligible_cards(cards, reserved=()):
    return sorted((c for c in cards if c['uuid'] not in reserved and not c['pids']
                   and c['utilization'] <= 5 and c['memory_free_mib'] >= MINIMUM_FREE_MIB),
                  key=lambda c: (-c['memory_free_mib'], c['index']))


def folder(root, batch):
    return root/'artifacts/gencdr_baseline'/validate_batch_id(batch)


def job_folder(out, job):
    return out/'jobs'/job['pair']/f"seed{job['seed']}"


def source_identity():
    source = git_identity(ROOT)
    if source['status_porcelain']:
        raise RuntimeError('Commit a clean checkout before starting production')
    paths = [Path(__file__).resolve(), ROOT/'requirements.txt']
    paths += sorted((ROOT/'cdr_framework/baselines/gencdr').glob('*.py'))
    return dict(commit=source['commit'], files={str(p.relative_to(ROOT)): sha(p) for p in paths},
                upstream='https://github.com/hupeiyu21/GenCDR', upstream_revision=UPSTREAM_REVISION)


def config_from(path):
    payload = read(path) if path else {}
    return BaselineConfig(**payload)


def local_model_seal(config):
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(config.model, revision=config.revision, local_files_only=True))
    if path.name != config.revision:
        raise RuntimeError('Backbone cache is not pinned')
    # Check actual weights as well as metadata; never download/modify a model cache.
    names = sorted(p for p in path.rglob('*') if p.is_file() and
        (p.suffix in ('.safetensors', '.bin') or p.name in ('config.json', 'model.safetensors.index.json')))
    if not names or not any(p.suffix in ('.safetensors', '.bin') for p in names):
        raise RuntimeError('Pinned backbone weights missing from existing cache')
    return dict(model=config.model, revision=config.revision,
                files={str(p.relative_to(path)): sha(p) for p in names})


def request(args, config):
    return dict(schema='gencdr-existing-assets-v1', model_label=MODEL_LABEL, paper_reproduction=False,
        root=str(args.root), source_batch=args.source_batch, jobs=jobs(), config=config.payload(),
        source=source_identity(), maximum_workers=args.max_workers, minimum_disk_gib=MINIMUM_DISK_GIB,
        minimum_free_mib=MINIMUM_FREE_MIB, job_ram_gib=24, automatic_empty_gpu_only=True,
        model_specific_tokenizer_fitting=True, processed_asset_writes=False,
        reconstructed_components=['item VIB router', 'universal expert mixer', 'recommendation decoder', 'user VIB router', 'prefix-tree decoding'],
        deviations=['existing normalized Qwen embeddings, no OpenAI/PCA', 'Qwen3-0.6B default, not paper Qwen2.5-7B',
            'existing shared-user samples, not author populations', 'tokenizer MTM not published/not implemented',
            'universal pretraining on both existing directional training files only',
            'existing histories concatenated with domain markers; no invented cross-domain ordering',
            'equal collision mass; stable item-ID ties', 'validation mean NDCG5/10, not paper Recall10'])


def identity(declaration, job):
    pair_assets = dict(declaration['assets'][job['pair']])
    pair_assets.update(declaration['assets'][REVERSE[job['pair']]])
    return dict(job=job, assets=pair_assets, source=declaration['request']['source'],
                config=declaration['request']['config'], backbone=declaration['backbone'],
                source_batch=declaration['request']['source_batch'], model_label=MODEL_LABEL)


def trained(out, job, declaration):
    path = job_folder(out, job)/'manifest.json'
    if not path.is_file():
        return False
    value = read(path)
    if (not value.get('complete') or value['identity'] != identity(declaration, job)
            or value['checkpoint_sha256'] != sha(path.parent/'best.pt')
            or value['tokenizer_sha256'] != sha(path.parent/'tokenizer.pt')):
        raise RuntimeError('Completed baseline output changed')
    return True


def host_admission(root, reserved=0):
    from experiments.run_gencdr_six import check_host_pressure, admit_host_budget
    if shutil.disk_usage(root).free < MINIMUM_DISK_GIB*2**30:
        raise ResourceWait(f'disk free below {MINIMUM_DISK_GIB} GiB')
    try:
        check_host_pressure()
        admit_host_budget(available=host_resources()['memory'].get('MemAvailable', 0), reserved=reserved, required=24*2**30)
    except RuntimeError as error:
        raise ResourceWait(str(error)) from error


def queue(args, out, declaration, phase, complete):
    pending = [j for j in jobs() if not complete(j)]
    active, failure = {}, None
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        while pending or active:
            for future in [f for f in active if f.done()]:
                job, card = active.pop(future)
                try:
                    future.result()
                    if not complete(job):
                        raise RuntimeError('Worker returned without complete evidence')
                    state = dict(state='complete', job=job, gpu_uuid=card['uuid'])
                except ResourceWait as error:
                    pending.append(job)
                    state = dict(state='waiting', job=job, error=str(error))
                except Exception as error:
                    failure = str(error)
                    state = dict(state='failed', job=job, error=failure)
                write_json_atomic(out/'status'/phase/f'{key(job)}.json', state)
            if failure:
                if active:
                    wait(active, return_when=FIRST_COMPLETED)
                    continue
                raise RuntimeError(f'Baseline worker failed; siblings drained: {failure}')
            reason = 'no empty GPU with >=24 GiB free'
            while pending and len(active) < args.max_workers:
                try:
                    host_admission(args.root, len(active)*24*2**30)
                    cards = eligible_cards(query_gpu_inventory(), {c['uuid'] for _, c in active.values()})
                except (ResourceWait, RuntimeError) as error:
                    reason = str(error)
                    break
                if not cards:
                    break
                match = None
                for j in pending:
                    runtime = job_folder(out, j)/'runtime.json'
                    affinity = read(runtime)['gpu_uuid'] if runtime.exists() else None
                    card = next((c for c in cards if affinity is None or c['uuid'] == affinity), None)
                    if card:
                        match = j, card
                        break
                if match is None:
                    reason = 'waiting for recorded GPU UUID affinities'
                    break
                job, card = match
                pending.remove(job)
                print(f'START {phase} {key(job)} GPU={card["index"]} {card["uuid"]}', flush=True)
                write_json_atomic(out/'status'/phase/f'{key(job)}.json', dict(state='running', job=job, gpu_uuid=card['uuid']))
                active[pool.submit(launch, args, out, job, card, phase)] = job, card
            if active:
                wait(active, timeout=args.poll_seconds, return_when=FIRST_COMPLETED)
            elif pending:
                print(f'WAIT_RESOURCE {reason}; pending={len(pending)}; recheck={args.poll_seconds}s', flush=True)
                time.sleep(args.poll_seconds)


def launch(args, out, job, card, phase):
    fresh = eligible_cards(query_gpu_inventory())
    if not any(c['uuid'] == card['uuid'] for c in fresh):
        raise ResourceWait('GPU occupancy changed before launch')
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(args.root),
        '--batch-id', args.batch_id, '--source-batch', args.source_batch, '--worker', phase,
        '--pair', job['pair'], '--seed', str(job['seed'])]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=card['uuid'], SIR_EXPECTED_GPU_UUID=card['uuid'])
    log = out/'logs'/f'{key(job)}_{phase}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode == 75:
        raise ResourceWait('GPU unavailable in child admission')
    if result.returncode:
        raise RuntimeError(f'Child exit {result.returncode}; {log}')


def worker(args):
    import torch
    out = folder(args.root, args.batch_id)
    declaration = read(out/'declaration.json')
    if declaration['request']['source'] != source_identity():
        raise RuntimeError('Worker code identity changed')
    config = BaselineConfig(**declaration['request']['config'])
    if local_model_seal(config) != declaration['backbone']:
        raise RuntimeError('Backbone weights changed')
    admitted = os.environ.get('SIR_EXPECTED_GPU_UUID')
    if not admitted or os.environ.get('CUDA_VISIBLE_DEVICES') != admitted:
        raise RuntimeError('Worker requires scheduler-admitted GPU UUID')
    # Query before allocating a CUDA context. Busy cards are retried, never shared.
    if not any(c['uuid'] == admitted for c in eligible_cards(query_gpu_inventory())):
        raise ResourceWait('GPU occupancy changed before CUDA initialization')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    job = dict(pair=args.pair, seed=args.seed)
    run = job_folder(out, job)
    with operation_lock(run/'operation'):
        bound = identity(declaration, job)
        verify_files(args.root, bound['assets'])
        device = torch.device('cuda:0')
        runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu_uuid=admitted,
                       gpu=torch.cuda.get_device_name(device), config=config.payload())
        # Runtime affinity is recorded for exact continuation; moving a partially
        # trained job to another card requires an independent experiment namespace.
        bind(run/'runtime.json', runtime)
        if args.worker == 'train':
            train_job(args.root, args.pair, args.seed, run, config, bound, device)
        else:
            freeze = read(out/'validation_freeze.json')
            if freeze['declaration_sha256'] != sha(out/'declaration.json'):
                raise RuntimeError('Batch freeze declaration changed')
            selected = next(j for j in freeze['jobs'] if j['job'] == job)
            evaluate_job(args.root, args.pair, args.seed, run, config, bound, device, selected)


def run(args):
    out = folder(args.root, args.batch_id)
    config = config_from(args.config)
    with operation_lock(out/'controller'):
        bind(out/'request.json', request(args, config))
        while True:
            completion = sir_completion(args.root, args.source_batch)
            if completion is not None:
                break
            print(f'WAIT_SIR {args.source_batch}: all 12 verified tests/results required; recheck={args.poll_seconds}s', flush=True)
            time.sleep(args.poll_seconds)
        print('SIR_COMPLETE_VERIFIED; binding existing assets', flush=True)
        seals = {p: ExistingAssets(args.root, p).seal() for p in PAIRS}
        declaration = bind(out/'declaration.json', dict(request=read(out/'request.json'), sir_completion=completion,
            assets=seals, backbone=local_model_seal(config)))
        queue(args, out, declaration, 'train', lambda j: trained(out, j, declaration))
        frozen = bind(out/'validation_freeze.json', dict(declaration_sha256=sha(out/'declaration.json'),
            jobs=[dict(job=j, identity=identity(declaration, j),
                       checkpoint_sha256=sha(job_folder(out, j)/'best.pt')) for j in jobs()], test_released=True))
        print('BASELINE_VALIDATION_FROZEN 12/12', flush=True)

        def test_done(job):
            path = job_folder(out, job)/'test.json'
            if not path.exists():
                return False
            report = read(path)
            selected = next(j for j in frozen['jobs'] if j['job'] == job)
            if (not report.get('complete') or report.get('split') != 'test' or report['freeze'] != selected
                    or report['identity'] != identity(declaration, job)):
                raise RuntimeError('Test report binding changed')
            bind(out/'test_completions'/f'{key(job)}.json', dict(job=job, report_sha256=sha(path),
                 freeze_sha256=sha(out/'validation_freeze.json')))
            return True
        queue(args, out, declaration, 'test', test_done)
        rows = [dict(**j, metrics=read(job_folder(out, j)/'test.json')['metrics']) for j in jobs()]
        for seals in declaration['assets'].values():
            verify_files(args.root, seals)
        bind(out/'results.json', dict(model_label=MODEL_LABEL, paper_reproduction=False,
            upstream_revision=UPSTREAM_REVISION, freeze_sha256=sha(out/'validation_freeze.json'),
            per_seed=rows, directions=summarize(rows), deviations=declaration['request']['deviations']))
        print(f'GENCDR_BASELINE_COMPLETE {out / "results.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/Data/lihb/SIR-CDR'))
    parser.add_argument('--batch-id', default=BASELINE_BATCH)
    parser.add_argument('--source-batch', default=SOURCE_BATCH)
    parser.add_argument('--max-workers', type=int, default=2)
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--action', choices=('run', 'status'), default='run')
    parser.add_argument('--worker', choices=('train', 'test'))
    parser.add_argument('--pair', choices=PAIRS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    args.root = args.root.resolve()
    validate_batch_id(args.batch_id)
    validate_batch_id(args.source_batch)
    if not 1 <= args.max_workers <= 8 or args.poll_seconds < 5:
        parser.error('max-workers 1..8 and poll-seconds >=5 required')
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', NVIDIA_TF32_OVERRIDE='0')
    out = folder(args.root, args.batch_id)
    if args.action == 'status':
        print(json.dumps(dict(model_label=MODEL_LABEL,
            request_exists=(out/'request.json').exists(), assets_bound=(out/'declaration.json').exists(),
            training_complete=sum((job_folder(out, j)/'manifest.json').is_file() and
                read(job_folder(out, j)/'manifest.json').get('complete') is True for j in jobs()),
            total_jobs=len(jobs()), frozen=(out/'validation_freeze.json').is_file(),
            test_complete=len(list((out/'test_completions').glob('*.json'))),
            summary_exists=(out/'results.json').is_file()), indent=2))
        return
    try:
        if args.worker:
            if not args.pair or args.seed is None:
                parser.error('Worker requires pair and seed')
            worker(args)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            run(args)
    except ResourceWait as error:
        print(f'WAIT_RESOURCE {error}', flush=True)
        raise SystemExit(75)


if __name__ == '__main__':
    main()
