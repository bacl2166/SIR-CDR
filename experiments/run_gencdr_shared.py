"""Isolated shared-data baseline workflow: convert -> assets -> ready -> run.

Existing forward/reverse experiment entry points and their identities are intact.
Use THIS entry point for the new protocol; old entry points keep the old data.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cdr_framework.datasets.gencdr_protocol import SCHEMA, build_shared, verify_shared, sha, write
from cdr_framework.datasets.gencdr_inputs import PAIRS, raw_amazon, raw_douban, load_bundle


DEFAULT_PAIRS = ('sports_to_clothing', 'phones_to_electronics', 'douban_book_movie')


def paths(root, pair):
    root = Path(root)
    return {'processed': root / 'data/processed' / SCHEMA / pair,
            'embedding': root / 'artifacts/embeddings' / SCHEMA / pair,
            'tokenizer': root / 'artifacts/tokenizer' / SCHEMA / pair,
            'control': root / 'artifacts/shared_protocol' / SCHEMA / pair,
            'runtime': root / 'configs/runtime' / SCHEMA / f'{pair}.yaml'}


def convert(root, pair, bundle=None, training_examples='all-prefixes'):
    source, target = PAIRS[pair]; p = paths(root, pair)
    if bundle:
        events, items, provenance = load_bundle(bundle, source, target)
    elif source in ('Book', 'Movie'):
        events, items, provenance = raw_douban(root)
    else:
        events, items, provenance = raw_amazon(root, source, target)
    return build_shared(events, items, source, target, p['processed'], provenance=provenance,
                        training_examples=training_examples)


def model_config(root, pair, seed):
    from cdr_framework.llm_config import LLMCDRConfig
    p = paths(root, pair)
    # Unified declared defaults, independent of previous pair-specific tuning.
    return LLMCDRConfig(processed_dir=p['processed'].resolve(), tokenizer_dir=p['tokenizer'].resolve(),
        output_dir=(Path(root) / 'artifacts/llm_cdr_v1' / SCHEMA / pair / f'seed{seed}').resolve(),
        llm_model_name='Qwen/Qwen3-0.6B', llm_revision='c1899de289a04d12100db370d81485cdf75e47ca',
        llm_gradient_checkpointing=False, batch_size=8, evaluation_batch_size=8,
        gradient_accumulation_steps=2, selection_metric='HR@10',
        evaluation_every=1, max_epochs=30, patience=5, scheduler_patience=2,
        top_ks=(5, 10, 20), max_sequence_length=50, hidden_dim=128,
        inference_mode='hybrid', retrieval_score_weight=1., generation_score_weight=.5,
        fusion_normalization='none', rerank_candidates=500, decode_chunk_size=128, seed=seed)


def runtime_path(root, pair, seed):
    p = paths(root, pair)['runtime']
    return p.with_name(f'{pair}_seed{seed}.yaml')


def asset_identity(root, pair):
    p = paths(root, pair)
    return {'protocol_sha256': sha(p['processed'] / 'shared_manifest.json'),
            'driver_sha256': sha(__file__), 'input_reader_sha256': sha(ROOT / 'cdr_framework/datasets/gencdr_inputs.py')}


def embedding(root, pair, device):
    from huggingface_hub import snapshot_download
    from cdr_framework.embeddings import Qwen3EmbeddingProvider, load_item_texts, run_embedding_job
    p = paths(root, pair); receipt = p['control'] / 'embedding_model.json'
    identity = asset_identity(root, pair)
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding='utf-8'))
        if old['identity'] != identity: raise RuntimeError('Shared embedding provenance changed')
        revision = old['revision']
    else:
        revision = None
    snapshot = Path(snapshot_download('Qwen/Qwen3-Embedding-8B', revision=revision or 'main'))
    import re
    if not re.fullmatch('[0-9a-f]{40}', snapshot.name) or (revision and snapshot.name != revision):
        raise RuntimeError('Embedding revision is not pinned')
    p['control'].mkdir(parents=True, exist_ok=True)
    if not receipt.exists():
        write(receipt, {'identity': identity, 'revision': snapshot.name, 'model_config_sha256': sha(snapshot/'config.json')})
    elif old['model_config_sha256'] != sha(snapshot/'config.json'):
        raise RuntimeError('Pinned embedding model config changed')
    os.environ['QWEN3_SHARED_EMBEDDING_PATH'] = str(snapshot)
    provider = Qwen3EmbeddingProvider(model='Qwen/Qwen3-Embedding-8B', dim=768, batch_size=2,
        device=device, max_sequence_length=512, model_path_env='QWEN3_SHARED_EMBEDDING_PATH')
    texts = p['processed'] / 'item_texts.jsonl'
    run_embedding_job(load_item_texts(texts), provider, p['embedding'], input_sha256=sha(texts),
        model='Qwen/Qwen3-Embedding-8B', dimension=768, batch_size=2, max_retries=5,
        retry_initial_seconds=2., force=False, identity_metadata={**identity,
            'model_revision': snapshot.name, 'provider': 'qwen3_local', 'normalized': True,
            'max_sequence_length': 512, 'model_config_sha256': sha(snapshot/'config.json')})


def tokenize(root, pair, device):
    from cdr_framework.experiment_config import TokenizerTrainingConfig
    from experiments.train_tokenizer import run
    p = paths(root, pair)
    config = TokenizerTrainingConfig(embeddings_path=p['embedding'] / 'text_embeddings.pt',
                                    item_texts_path=p['processed'] / 'item_texts.jsonl', output_dir=p['tokenizer'])
    from dataclasses import asdict
    execution = {'identity': asset_identity(root, pair), 'device': device,
                 'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
                 'embedding_sha256': sha(config.embeddings_path), 'texts_sha256': sha(config.item_texts_path),
                 'domains': list(PAIRS[pair])}
    receipt = p['control']/'tokenizer_execution.json'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')) != execution:
        raise RuntimeError('Shared tokenizer recipe/input/device changed; do not resume or reuse it')
    p['control'].mkdir(parents=True, exist_ok=True)
    if not receipt.exists(): write(receipt, execution)
    run(config, device=device, force=False, source_domain=PAIRS[pair][0], target_domain=PAIRS[pair][1])
    if json.loads((p['tokenizer']/'quality_report.json').read_text())['passed'] is not True:
        raise RuntimeError('Shared tokenizer quality gate failed')


def fingerprint_files(root, pair):
    p = paths(root, pair)
    required = [p['processed'] / 'shared_manifest.json', p['embedding'] / 'text_embeddings.pt',
                p['embedding'] / 'manifest.json', p['control'] / 'embedding_model.json']
    required += [p['control']/'tokenizer_execution.json']
    required += [p['tokenizer'] / name for name in ('item_latents.pt', 'semantic_ids.pt', 'tokenizer.pt',
                                                  'quality_report.json', 'manifest.json')]
    return {str(file.relative_to(root)).replace('\\', '/'): sha(file) for file in required}


def validate_asset_chain(root, pair):
    p = paths(root, pair)
    text_hash = sha(p['processed']/'item_texts.jsonl')
    emb = json.loads((p['embedding']/'manifest.json').read_text(encoding='utf-8'))
    if emb.get('input_sha256') != text_hash:
        raise RuntimeError('Shared embedding belongs to different item texts')
    if emb.get('protocol_sha256') != sha(p['processed']/'shared_manifest.json'):
        raise RuntimeError('Shared embedding belongs to a different protocol')
    token = json.loads((p['tokenizer']/'manifest.json').read_text(encoding='utf-8'))
    if (token.get('embeddings_sha256') != sha(p['embedding']/'text_embeddings.pt')
            or token.get('item_texts_sha256') != text_hash or token.get('domain_labels') != list(PAIRS[pair])):
        raise RuntimeError('Tokenizer was trained on different vectors/texts/domain roles')
    execution = json.loads((p['control']/'tokenizer_execution.json').read_text(encoding='utf-8'))
    if execution['identity'] != asset_identity(root, pair) or execution['embedding_sha256'] != token['embeddings_sha256']:
        raise RuntimeError('Tokenizer execution receipt differs')


def seal(root, pair):
    p = paths(root, pair); verify_shared(p['processed'])
    validate_asset_chain(root, pair)
    from cdr_framework.formal_training import load_fixed_catalog
    catalog = load_fixed_catalog(model_config(root, pair, 42))
    mapping = json.loads((p['processed']/'mappings.json').read_text(encoding='utf-8'))
    if len(catalog.item_latents) != len(mapping['id_to_item']):
        raise RuntimeError('Shared tokenizer rows and item mapping differ')
    value = {'identity': asset_identity(root, pair), 'files': fingerprint_files(root, pair), 'complete': True}
    destination = p['control'] / 'ready.json'
    if destination.exists() and json.loads(destination.read_text(encoding='utf-8')) != value:
        raise RuntimeError('Shared asset seal differs; no overwrite or reuse of mixed assets')
    write(destination, value)


def verify_ready(root, pair):
    p = paths(root, pair); verify_shared(p['processed'])
    validate_asset_chain(root, pair)
    value = json.loads((p['control']/'ready.json').read_text(encoding='utf-8'))
    if value != {'identity': asset_identity(root, pair), 'files': fingerprint_files(root, pair), 'complete': True}:
        raise RuntimeError('Shared training assets changed; refusing training')
    return True


def execute(command):
    print('$ '+' '.join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    subprocess.run(list(map(str, command)), cwd=ROOT, env=environment, check=True)


def workflow(root, pair, action, seed, device, bundle=None, training_examples='all-prefixes'):
    from experiments.recover_llm_evaluation import operation_lock
    p = paths(root, pair)
    if action == 'status':
        result = {'pair': pair, 'processed': (p['processed']/'shared_manifest.json').exists(),
                  'embedding': (p['embedding']/'text_embeddings.pt').exists(),
                  'tokenizer': (p['tokenizer']/'quality_report.json').exists(),
                  'ready': (p['control']/'ready.json').exists()}
        print(json.dumps(result), flush=True); return
    with operation_lock(p['control']/'controller'):
        if action in ('convert', 'all'):
            convert(root, pair, bundle, training_examples)
        else:
            verify_shared(p['processed'])
        if action == 'convert': return
        if action in ('embedding', 'tokenize'):
            from experiments.prepare_reverse_assets import ensure_gpu_idle
            if device.startswith('cuda'): ensure_gpu_idle()
            (embedding if action == 'embedding' else tokenize)(root, pair, device)
            return
        if action in ('prepare', 'all'):
            from experiments.prepare_reverse_assets import ensure_gpu_idle
            if device.startswith('cuda'): ensure_gpu_idle()
            for stage in ('embedding', 'tokenize'):
                execute([sys.executable, __file__, '--action', stage, '--pairs', pair, '--root', root, '--device', device,
                         '--asset-worker'])
            seal(root, pair)
            from experiments.run_llm_cdr import prepare_model, preflight
            prepare_model(model_config(root, pair, seed), runtime_path(root, pair, seed))
            preflight(model_config(root, pair, seed))
        if action == 'preflight':
            verify_ready(root, pair)
            from experiments.run_llm_cdr import preflight
            preflight(model_config(root, pair, seed)); return
        if action in ('run', 'all'):
            from experiments.prepare_reverse_assets import ensure_gpu_idle
            if device.startswith('cuda'): ensure_gpu_idle()
            verify_ready(root, pair)
            from experiments.run_llm_cdr import absolute_config
            from cdr_framework.llm_training import config_payload
            runtime = runtime_path(root, pair, seed)
            if config_payload(absolute_config(runtime)) != config_payload(model_config(root, pair, seed)):
                raise RuntimeError('Runtime points to a different data recipe, seed, or output')
            execute([sys.executable, ROOT/'experiments/run_llm_cdr.py', '--action', 'train', '--config', runtime,
                     '--device', device])
            tuning = Path(root)/'artifacts/tuning'/SCHEMA/pair/f'seed{seed}'
            execute([sys.executable, ROOT/'experiments/recover_llm_evaluation.py', '--action', 'all',
                     '--config', runtime, '--output', tuning, '--device', device])
    print(f'=== SHARED {pair} {action.upper()} COMPLETED ===', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('convert', 'prepare', 'preflight', 'run', 'all', 'status', 'embedding', 'tokenize'), required=True)
    parser.add_argument('--pairs', nargs='+', choices=tuple(PAIRS), default=list(DEFAULT_PAIRS))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--training-examples', choices=('all-prefixes', 'last-prefix'), default='all-prefixes')
    parser.add_argument('--bundle', type=Path, help='Normalized timestamped external sequence bundle; requires one pair')
    parser.add_argument('--root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument('--asset-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.bundle and len(args.pairs) != 1: parser.error('--bundle requires exactly one pair')
    if len(args.pairs) != len(set(args.pairs)): parser.error('Duplicate tasks')
    if args.asset_worker:
        if args.action not in ('embedding', 'tokenize') or len(args.pairs) != 1: parser.error('Invalid worker')
        # Parent holds the controller lock. Each heavy stage has a fresh process,
        # releasing the 8B model before tokenizer/recommender allocation.
        pair = args.pairs[0]; verify_shared(paths(args.root, pair)['processed'])
        (embedding if args.action == 'embedding' else tokenize)(args.root, pair, args.device)
        return
    for pair in args.pairs:
        workflow(args.root.resolve(), pair, args.action, args.seed, args.device, args.bundle, args.training_examples)


if __name__ == '__main__': main()
