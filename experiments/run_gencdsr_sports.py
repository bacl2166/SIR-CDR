"""Explicit, isolated stages for the reconstructed GenCDSR Sports/Clothing protocol.

Each invocation runs one stage. No scheduler or GPU selection is implicit.
"""
import argparse
import csv
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
SCHEMA = 'gencdsr-inspired-sc-v2'
PAIRS = ('sports_to_clothing', 'clothing_to_sports')
SEEDS = (42, 43, 44)
LLM_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
EMBEDDING_REVISION = '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'
GPU_ACTIONS = ('embedding', 'tokenize', 'smoke', 'train', 'validate', 'test')
ACTIONS = ('plan', 'convert', 'audit', 'embedding', 'tokenize', 'seal', 'runtime',
           'smoke', 'train', 'validate', 'freeze', 'test', 'status')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def immutable(path, value):
    """Publish one complete JSON document atomically, never replace a receipt."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        try: os.link(temporary, path)
        except FileExistsError:
            if read(path) != json.loads(data): raise RuntimeError(f'Immutable receipt differs: {path}')
    finally: os.unlink(temporary)
    return value


@contextmanager
def lock(path):
    """Kernel lock released on process exit; lock files never enter model output."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, 2)
            if not stream.tell(): stream.write(b'0'); stream.flush()
            stream.seek(0)
            try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error: raise RuntimeError(f'Operation busy: {path}') from error
            try: yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error: raise RuntimeError(f'Operation busy: {path}') from error
            try: yield
            finally: fcntl.flock(stream, fcntl.LOCK_UN)


def paths(root, pair, seed=42, study='sports-v1'):
    if pair not in PAIRS or seed not in SEEDS: raise ValueError('Unsupported pair or seed')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', study): raise ValueError('Invalid study ID')
    root = Path(root).resolve()
    control = root/'artifacts/experiment_control'/SCHEMA/study
    return {'processed': root/'data/processed'/SCHEMA/pair,
            'embedding': root/'artifacts/embeddings'/SCHEMA/pair,
            'tokenizer': root/'artifacts/tokenizer'/SCHEMA/pair,
            'assets': root/'artifacts/shared_protocol'/SCHEMA/pair,
            'control': control, 'job': control/pair/f'seed{seed}',
            'runtime': root/'configs/runtime'/SCHEMA/study/f'{pair}_seed{seed}.yaml',
            'output': root/'artifacts/llm_cdr_v1'/SCHEMA/study/pair/f'seed{seed}'}


def model_config(root, pair, seed, study):
    from cdr_framework.llm_config import LLMCDRConfig
    p = paths(root, pair, seed, study)
    return LLMCDRConfig(processed_dir=p['processed'], tokenizer_dir=p['tokenizer'], output_dir=p['output'],
        llm_model_name='Qwen/Qwen3-0.6B', llm_revision=LLM_REVISION,
        llm_gradient_checkpointing=False, batch_size=8, gradient_accumulation_steps=2,
        evaluation_batch_size=8, selection_metric='NDCG@10', evaluation_every=1,
        max_epochs=30, patience=5, scheduler_patience=2, top_ks=(5,10,20),
        max_sequence_length=50, hidden_dim=128, inference_mode='hybrid',
        retrieval_score_weight=1., generation_score_weight=.5, fusion_normalization='none',
        rerank_candidates=500, decode_chunk_size=128, seed=seed)


def payload(config):
    return {k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
            for k,v in asdict(config).items()}


def source_identity():
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    branch = git('branch', '--show-current')
    if branch in ('main', 'master') or git('status', '--porcelain'):
        raise RuntimeError('Run committed, clean experiment checkout outside main/master')
    files = git('ls-files', '*.py').splitlines()
    return {'commit':git('rev-parse', 'HEAD'), 'branch':branch,
            'files':{name:sha(ROOT/name) for name in files}}


def asset_identity(root, pair):
    p = paths(root, pair)
    return {'schema':SCHEMA, 'source':source_identity(),
            'protocol_sha256':sha(p['processed']/'shared_manifest.json')}


def plan(root, data_root, study):
    return {'schema':SCHEMA, 'root':str(Path(root).resolve()), 'data_root':str(Path(data_root).resolve()),
            'study':study, 'pairs':list(PAIRS), 'seeds':list(SEEDS), 'pilot_seed':42,
            'automatic_launch':False, 'paper_alignment_verified':False,
            'filter':'independent full-domain item>=3/user>=5 fixed points, then common users; no common-subset item re-pruning',
            'ranking':'full target catalog; full-seen masking; predeclared hybrid500/0.5/none',
            'selection_metric':'NDCG@10', 'llm_revision':LLM_REVISION,
            'embedding_revision':EMBEDDING_REVISION, 'actions':list(ACTIONS)}


def admit_gpu(uuid, inventory, host_available):
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid or ''):
        raise ValueError('--gpu requires a full physical GPU UUID')
    cards = [c for c in inventory if c['uuid'] == uuid]
    if len(cards) != 1: raise RuntimeError('GPU absent or ambiguous')
    card = cards[0]; free = card.get('free_mib', card.get('memory_free_mib'))
    if (card['pids'] or not math.isfinite(card['utilization']) or card['utilization'] != 0
            or free is None or not math.isfinite(free) or free < 12288 or host_available < 24*2**30):
        raise RuntimeError('GPU must be idle, process-free, >=12GiB free; host requires >=24GiB available')
    return card


def affinity(job, uuid):
    return immutable(Path(job)/'gpu_affinity.json', {'uuid':uuid})


def physical_resources():
    """Only stdlib and nvidia-smi: importing the framework itself loads torch."""
    def query(kind,fields):
        output = subprocess.run(['nvidia-smi',f'--query-{kind}={fields}','--format=csv,noheader,nounits'],
            check=True,capture_output=True,text=True,timeout=20).stdout
        return list(csv.reader(output.strip().splitlines(),skipinitialspace=True))
    try:
        cards = [{'uuid':u,'free_mib':float(f),'utilization':float(v),'pids':[]}
                 for u,f,v in query('gpu','uuid,memory.free,utilization.gpu')]
        by_uuid = {c['uuid']:c for c in cards}
        if not cards or len(by_uuid) != len(cards): raise ValueError('Ambiguous GPU inventory')
        for uuid,pid in query('compute-apps','gpu_uuid,pid'): by_uuid[uuid]['pids'].append(int(pid))
        memory = {line.split(':')[0]:int(line.split(':')[1].strip().split()[0])*1024
                  for line in Path('/proc/meminfo').read_text().splitlines()}
        return cards,memory['MemAvailable']
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
        raise RuntimeError('Cannot establish physical GPU occupancy and host memory; refusing CUDA') from error


@contextmanager
def gpu_session(uuid, job, sticky=True):
    # Validate spelling before deriving any filesystem path or importing torch.
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid or ''):
        raise ValueError('--gpu requires a full physical GPU UUID')
    with lock(Path(tempfile.gettempdir())/'sir-cdr-physical-gpus'/f'{uuid}.lock'):
        inventory,host = physical_resources()
        card = admit_gpu(uuid, inventory, host)
        if sticky: affinity(job, uuid)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                          TOKENIZERS_PARALLELISM='false')
        import torch
        if torch.cuda.device_count() != 1: raise RuntimeError('Expected exactly one visible GPU')
        actual = str(getattr(torch.cuda.get_device_properties(0), 'uuid', ''))
        if actual.lower().removeprefix('gpu-') != uuid.lower().removeprefix('gpu-'):
            raise RuntimeError('Logical CUDA UUID differs from physical admission')
        torch.set_num_threads(1)
        yield torch.device('cuda:0'), card


def embedding(root, pair, device):
    from huggingface_hub import snapshot_download
    from cdr_framework.embeddings import Qwen3EmbeddingProvider, load_item_texts, run_embedding_job
    p = paths(root, pair); identity = asset_identity(root, pair)
    snapshot = Path(snapshot_download('Qwen/Qwen3-Embedding-8B', revision=EMBEDDING_REVISION))
    if snapshot.name != EMBEDDING_REVISION: raise RuntimeError('Embedding snapshot revision mismatch')
    model = {'identity':identity, 'revision':snapshot.name, 'model_config_sha256':sha(snapshot/'config.json')}
    immutable(p['assets']/'embedding_model.json', model)
    os.environ['SIR_GENCDSR_EMBEDDING_PATH'] = str(snapshot)
    provider = Qwen3EmbeddingProvider(model='Qwen/Qwen3-Embedding-8B', dim=768, batch_size=2,
        device=str(device), max_sequence_length=512, model_path_env='SIR_GENCDSR_EMBEDDING_PATH')
    texts = p['processed']/'item_texts.jsonl'
    run_embedding_job(load_item_texts(texts), provider, p['embedding'], input_sha256=sha(texts),
        model='Qwen/Qwen3-Embedding-8B', dimension=768, batch_size=2, max_retries=5,
        retry_initial_seconds=2., force=False, identity_metadata={**identity,
            'model_revision':snapshot.name, 'provider':'qwen3_local', 'normalized':True,
            'max_sequence_length':512, 'model_config_sha256':model['model_config_sha256']})


def tokenize(root, pair, device):
    from cdr_framework.experiment_config import TokenizerTrainingConfig
    from cdr_framework.datasets.gencdsr_protocol import DOMAINS
    from experiments.train_tokenizer import run
    p = paths(root, pair)
    config = TokenizerTrainingConfig(embeddings_path=p['embedding']/'text_embeddings.pt',
        item_texts_path=p['processed']/'item_texts.jsonl', output_dir=p['tokenizer'])
    immutable(p['assets']/'tokenizer_execution.json', {'identity':asset_identity(root,pair),
        'config':payload(config), 'embeddings_sha256':sha(config.embeddings_path),
        'texts_sha256':sha(config.item_texts_path), 'domains':list(DOMAINS[pair])})
    run(config, device=str(device), force=False, source_domain=DOMAINS[pair][0], target_domain=DOMAINS[pair][1])
    if read(p['tokenizer']/'quality_report.json')['passed'] is not True: raise RuntimeError('Tokenizer quality failed')


def asset_files(root, pair):
    p = paths(root, pair)
    files = [p['processed']/'shared_manifest.json']
    files += [p['processed']/name for name in sorted(read(p['processed']/'shared_manifest.json')['files'])]
    files += [p['embedding']/name for name in ('text_embeddings.pt','manifest.json')]
    files += [p['tokenizer']/name for name in ('item_latents.pt','semantic_ids.pt','tokenizer.pt','quality_report.json','manifest.json')]
    files += [p['assets']/'embedding_model.json', p['assets']/'tokenizer_execution.json']
    return {str(f.relative_to(Path(root).resolve())).replace('\\','/'):sha(f) for f in files}


def asset_chain(root, pair):
    from cdr_framework.experiment_control import require_embedding_recipe
    from cdr_framework.datasets.gencdsr_protocol import audit, DOMAINS
    p = paths(root,pair); audit(p['processed']); identity = asset_identity(root,pair)
    model = read(p['assets']/'embedding_model.json'); emb = read(p['embedding']/'manifest.json')
    texts = p['processed']/'item_texts.jsonl'; text_hash = sha(texts)
    if model['identity'] != identity or model['revision'] != EMBEDDING_REVISION: raise RuntimeError('Embedding source differs')
    if emb.get('protocol_sha256') != identity['protocol_sha256']: raise RuntimeError('Embedding protocol differs')
    with texts.open(encoding='utf-8') as stream: count = sum(bool(line.strip()) for line in stream)
    require_embedding_recipe(emb, {'input_sha256':text_hash, 'model_revision':EMBEDDING_REVISION,
        'model_config_sha256':model['model_config_sha256'], 'dimension':768,
        'normalized':True, 'max_sequence_length':512, 'item_count':count})
    token = read(p['tokenizer']/'manifest.json'); execution = read(p['assets']/'tokenizer_execution.json')
    if (token.get('embeddings_sha256') != sha(p['embedding']/'text_embeddings.pt')
            or token.get('item_texts_sha256') != text_hash or token.get('domain_labels') != list(DOMAINS[pair])
            or execution['identity'] != identity or execution['embeddings_sha256'] != token['embeddings_sha256']
            or execution['texts_sha256'] != text_hash or read(p['tokenizer']/'quality_report.json')['passed'] is not True):
        raise RuntimeError('Tokenizer source or quality differs')


def seal(root, pair):
    asset_chain(root,pair)
    from cdr_framework.formal_training import load_fixed_catalog
    load_fixed_catalog(model_config(root,pair,42,'asset-check'))
    return immutable(paths(root,pair)['assets']/'ready.json',
                     {'identity':asset_identity(root,pair), 'files':asset_files(root,pair), 'complete':True})


def verify_ready(root, pair):
    asset_chain(root,pair)
    value = read(paths(root,pair)['assets']/'ready.json')
    if value != {'identity':asset_identity(root,pair), 'files':asset_files(root,pair), 'complete':True}:
        raise RuntimeError('Sealed assets changed')
    return value


def runtime(root, pair, seed, study):
    from huggingface_hub import snapshot_download
    from experiments.run_llm_cdr import MODEL_FILES, _validate_snapshot
    cfg = model_config(root,pair,seed,study)
    snapshot = Path(snapshot_download(cfg.llm_model_name, revision=LLM_REVISION, allow_patterns=MODEL_FILES))
    if snapshot.name != LLM_REVISION: raise RuntimeError('LLM revision differs')
    _validate_snapshot(snapshot)
    p = paths(root,pair,seed,study)
    immutable(p['runtime'], {'llm_recommendation':payload(cfg)})
    immutable(p['job']/'runtime.json', {'source':source_identity(), 'assets':verify_ready(root,pair),
        'config_sha256':sha(p['runtime']), 'model_files':{f.name:sha(f) for f in sorted(snapshot.iterdir()) if f.is_file()}})
    return cfg


def verify_runtime(root,pair,seed,study):
    p = paths(root,pair,seed,study); cfg = model_config(root,pair,seed,study)
    receipt = read(p['job']/'runtime.json')
    if (read(p['runtime']) != {'llm_recommendation':payload(cfg)} or receipt['source'] != source_identity()
            or receipt['assets'] != verify_ready(root,pair) or receipt['config_sha256'] != sha(p['runtime'])):
        raise RuntimeError('Runtime config/source/assets changed')
    from experiments.run_llm_cdr import check_cached_model
    snapshot = check_cached_model(cfg)
    if receipt['model_files'] != {f.name:sha(f) for f in sorted(snapshot.iterdir()) if f.is_file()}:
        raise RuntimeError('Pinned model cache changed')
    return cfg


def completed(config):
    import torch
    from cdr_framework.llm_training import signature
    manifest = read(config.output_dir/'manifest.json')
    checkpoint = torch.load(config.output_dir/'best_model.pt', map_location='cpu', weights_only=True)
    last = torch.load(config.output_dir/'last_checkpoint.pt', map_location='cpu', weights_only=True)
    identity = signature(config)
    if (manifest.get('complete') is not True or manifest.get('identity') != identity
            or checkpoint.get('identity') != identity or checkpoint.get('validation') != manifest.get('best_validation')
            or last.get('identity') != identity or last.get('epoch') != manifest.get('epochs_completed')
            or not 1 <= checkpoint.get('epoch',0) <= manifest.get('epochs_completed',0) <= config.max_epochs
            or (manifest['epochs_completed'] < config.max_epochs and last.get('stale',0) < config.patience)
            or not math.isfinite(checkpoint.get('validation',{}).get(config.selection_metric,float('nan')))):
        raise RuntimeError('Completed manifest/checkpoint identity, metric or epoch mismatch')
    return checkpoint


def binding(root,pair,seed,study):
    p = paths(root,pair,seed,study); cfg = verify_runtime(root,pair,seed,study); checkpoint = completed(cfg)
    validation = read(p['output']/'validation_hybrid.json')
    if read(p['job']/'validation.json') != {'report_sha256':sha(p['output']/'validation_hybrid.json')}:
        raise RuntimeError('Validation receipt changed')
    if (validation.get('split') != 'validation' or validation.get('mode') != 'hybrid'
            or validation.get('epoch') != checkpoint['epoch']
            or validation.get('checkpoint_sha256') != sha(p['output']/'best_model.pt')
            or validation.get('metrics') != checkpoint['validation']):
        # Timing fields are measured again by evaluation, and are not model metrics.
        def metrics(value): return {k:v for k,v in value.items() if k != 'seconds'}
        if (validation.get('split') != 'validation' or validation.get('mode') != 'hybrid'
                or validation.get('epoch') != checkpoint['epoch']
                or validation.get('checkpoint_sha256') != sha(p['output']/'best_model.pt')
                or metrics(validation.get('metrics',{})) != metrics(checkpoint['validation'])):
            raise RuntimeError('Validation is not bound to selected checkpoint')
    from cdr_framework.text_training import evaluation_identity
    if validation.get('evaluation_identity') != evaluation_identity(cfg,p['output']/'best_model.pt','validation','hybrid'):
        raise RuntimeError('Validation evaluation identity differs')
    files = [p['runtime'], p['job']/'runtime.json', p['job']/'validation.json',
             p['output']/'manifest.json', p['output']/'best_model.pt', p['output']/'validation_hybrid.json']
    return {'pair':pair, 'seed':seed, 'source':source_identity(), 'assets':verify_ready(root,pair),
            'files':{str(f.relative_to(Path(root).resolve())).replace('\\','/'):sha(f) for f in files}}


def freeze_path(root,study):
    return paths(root,PAIRS[0],42,study)['control']/'freeze.json'


def audit_directions(root):
    from cdr_framework.datasets.gencdsr_protocol import audit_pair_roots
    return audit_pair_roots(*(paths(root,pair)['processed'] for pair in PAIRS))


def freeze(root,study):
    with ExitStack() as locks:
        for pair in PAIRS:
            for seed in SEEDS: locks.enter_context(lock(paths(root,pair,seed,study)['job']/'operation.lock'))
        audit_directions(root)
        jobs = []
        for pair in PAIRS:
            for seed in SEEDS:
                p = paths(root,pair,seed,study)
                if (p['output']/'test.json').exists() or (p['job']/'test-reservation.json').exists():
                    raise RuntimeError('Prior test access prohibits a new freeze')
                jobs.append(binding(root,pair,seed,study))
        return immutable(freeze_path(root,study), {'schema':SCHEMA, 'study':study, 'jobs':jobs,
            'policy':{'selection':'NDCG@10', 'ranking':'hybrid500/0.5/none', 'seeds':list(SEEDS)}})


def verify_freeze(root,study):
    frozen = read(freeze_path(root,study))
    expected = [(pair,seed) for pair in PAIRS for seed in SEEDS]
    if [(j.get('pair'),j.get('seed')) for j in frozen.get('jobs',[])] != expected:
        raise RuntimeError('Freeze requires both pairs and all three seeds')
    if frozen != {'schema':SCHEMA, 'study':study, 'jobs':[binding(root,p,s,study) for p,s in expected],
                  'policy':{'selection':'NDCG@10','ranking':'hybrid500/0.5/none','seeds':list(SEEDS)}}:
        raise RuntimeError('Frozen checkpoint/config/source/assets/validation changed')
    return frozen


def workflow(root,data_root,study,pair,seed,action,gpu):
    root = Path(root).resolve(); data_root = Path(data_root).resolve()
    pair = pair or PAIRS[0]; p = paths(root,pair,seed,study)
    if action == 'plan': return plan(root,data_root,study)
    if action == 'status':
        return {'freeze':freeze_path(root,study).exists(), 'pairs':{q:{k:str(v) for k,v in paths(root,q,seed,study).items()} for q in PAIRS}}
    if action == 'test': read(freeze_path(root,study))  # fail before CUDA and heavy imports
    with lock(p['control']/'study.lock') if action in ('freeze','test') else lock(p['job']/'operation.lock'):
        if action == 'freeze': return freeze(root,study)
        if action in ('convert','audit','embedding','tokenize','seal'):
            with lock(p['assets']/'operation.lock'):
                if action == 'convert':
                    from cdr_framework.datasets.gencdsr_protocol import convert
                    if root == data_root: raise RuntimeError('Raw checkout and new output checkout must differ')
                    source_identity(); return str(convert(data_root,pair,output_root=root))
                if action == 'audit':
                    from cdr_framework.datasets.gencdsr_protocol import audit
                    return audit(p['processed'], raw_root=data_root)
                if action == 'seal': return seal(root,pair)
                if (p['assets']/'ready.json').exists(): raise RuntimeError('Assets already sealed')
                with gpu_session(gpu,p['job'],sticky=False) as (device,_):
                    from cdr_framework.datasets.gencdsr_protocol import audit
                    audit(p['processed']); (embedding if action == 'embedding' else tokenize)(root,pair,device)
                return {'stage':action,'complete':True}
        if action == 'runtime': return payload(runtime(root,pair,seed,study))
        if action != 'test' and freeze_path(root,study).exists(): raise RuntimeError('Freeze closes training/validation')
        with gpu_session(gpu,p['job']) as (device,_):
            cfg = verify_runtime(root,pair,seed,study)
            from cdr_framework import llm_training
            if action == 'smoke':
                from experiments.run_llm_cdr import smoke
                return immutable(p['job']/'smoke.json', smoke(cfg,device))
            if action == 'train':
                if seed != 42:
                    for pilot_pair in PAIRS: binding(root,pilot_pair,42,study)
                if not (p['job']/'smoke.json').exists(): raise RuntimeError('Run smoke stage before training')
                return llm_training.train(cfg,device)
            completed(cfg)
            if action == 'validate':
                if (p['job']/'validation.json').exists(): return binding(root,pair,seed,study)
                result = llm_training.evaluate_checkpoint(cfg,device,'validation')
                immutable(p['job']/'validation.json', {'report_sha256':sha(p['output']/'validation_hybrid.json')})
                binding(root,pair,seed,study); return result
            frozen = verify_freeze(root,study); freeze_hash = sha(freeze_path(root,study))
            if (p['job']/'test-complete.json').exists():
                receipt = read(p['job']/'test-complete.json')
                if receipt != {'freeze_sha256':freeze_hash,'report_sha256':sha(p['output']/'test.json')}:
                    raise RuntimeError('Immutable test report changed')
                return read(p['output']/'test.json')
            reservation = p['job']/'test-reservation.json'
            if reservation.exists() or (p['output']/'test.json').exists(): raise RuntimeError('Test already reserved; no retry')
            immutable(reservation, {'freeze_sha256':freeze_hash, 'pair':pair,'seed':seed})
            result = llm_training.evaluate_checkpoint(cfg,device,'test')
            immutable(p['job']/'test-complete.json', {'freeze_sha256':freeze_hash,'report_sha256':sha(p['output']/'test.json')})
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=ACTIONS, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data-root', type=Path, default=Path('/Data/lihb/SIR-CDR'))
    parser.add_argument('--study-id', default='sports-clothing-gencdsr-v2')
    parser.add_argument('--pair', choices=PAIRS)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--gpu', help='Explicit full physical GPU UUID; required for every GPU stage')
    args = parser.parse_args()
    if not args.pair and args.action not in ('plan','convert','audit','freeze','status'): parser.error('--pair is required')
    if args.action in GPU_ACTIONS and not args.gpu: parser.error('--gpu full UUID is required')
    pairs = PAIRS if not args.pair and args.action in ('convert','audit') else (args.pair,)
    results = [workflow(args.root,args.data_root,args.study_id,p,args.seed,args.action,args.gpu) for p in pairs]
    if args.action == 'audit' and len(pairs) == 2:
        from cdr_framework.datasets.gencdsr_protocol import audit_pair_roots
        results.append(audit_pair_roots(*(paths(args.root,p)['processed'] for p in PAIRS)))
    print(json.dumps(results,indent=2,allow_nan=False))


if __name__ == '__main__': main()
