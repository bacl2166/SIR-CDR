"""Prepare and run the separate Douban Book -> Movie recommendation task."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdr_framework.datasets.douban import convert_archive, verify_processed, _identity
from experiments.prepare_gencdr_douban import download_archive, ARCHIVE_SHA256, sha256

ASSET_CONFIG = ROOT / 'configs/douban_book_movie.yaml'
MODEL_CONFIG = ROOT / 'configs/llm_douban_book_movie_qwen06_fast.yaml'
RUNTIME = ROOT / 'configs/runtime/llm_douban_book_movie_qwen06_fast.yaml'
PROCESSED = ROOT / 'data/processed/douban_book_movie'
ARCHIVE = ROOT / 'data/raw/douban_fengzhu_v1.zip'
TUNING = ROOT / 'artifacts/tuning/douban_book_movie_qwen06_fast_seed42_snapshot_eval_v1'


def commands(stage, device):
    python = sys.executable
    cli = [python, str(ROOT / 'experiments/run_llm_cdr.py')]
    if stage == 'prepare':
        return [[python, str(Path(__file__)), '--action', action, '--device', device]
                for action in ('embedding', 'tokenize')] + [
            cli + ['--action', 'prepare', '--config', str(MODEL_CONFIG), '--resolved-config', str(RUNTIME)],
            cli + ['--action', 'preflight', '--config', str(RUNTIME)]]
    if stage == 'run':
        return [cli + ['--action', 'train', '--config', str(RUNTIME), '--device', device],
                [python, str(ROOT / 'experiments/recover_llm_evaluation.py'), '--action', 'all',
                 '--config', str(RUNTIME), '--output', str(TUNING), '--device', device]]
    raise ValueError('Unknown workflow stage')


def embedding(device):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from cdr_framework.experiment_config import QwenEmbeddingConfig
    from cdr_framework.embeddings import Qwen3EmbeddingProvider, load_item_texts, run_embedding_job
    config = QwenEmbeddingConfig.from_yaml(ASSET_CONFIG)
    receipt = ROOT / 'artifacts/douban_book_movie/embedding_model.json'
    if receipt.exists():
        recorded = json.loads(receipt.read_text(encoding='utf-8'))
        if recorded['model'] != config.model:
            raise RuntimeError('Embedding model receipt differs')
        snapshot = Path(snapshot_download(config.model, revision=recorded['revision']))
    else:
        # Prefer the already downloaded 8B snapshot; persist its actual immutable
        # revision, not a moving "main", before generating any vectors.
        try:
            snapshot = Path(snapshot_download(config.model, local_files_only=True))
        except LocalEntryNotFoundError:
            snapshot = Path(snapshot_download(config.model))
        recorded = {'model': config.model, 'revision': snapshot.name}
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open('x', encoding='utf-8') as stream: json.dump(recorded, stream, indent=2)
    if len(recorded['revision']) != 40 or snapshot.name != recorded['revision']:
        raise RuntimeError('Embedding snapshot revision is not immutable')
    os.environ[config.model_path_env] = str(snapshot)
    provider = Qwen3EmbeddingProvider(model=config.model, dim=config.dimension, batch_size=config.batch_size,
        device=device, max_sequence_length=config.max_sequence_length, model_path_env=config.model_path_env)
    texts = ROOT / config.input_path
    result = run_embedding_job(load_item_texts(texts), provider, ROOT / config.output_dir,
        input_sha256=sha256(texts), model=config.model, dimension=config.dimension, batch_size=config.batch_size,
        max_retries=config.max_retries, retry_initial_seconds=config.retry_initial_seconds, force=False,
        identity_metadata={'provider': config.provider, 'normalized': True,
            'max_sequence_length': config.max_sequence_length, 'model_revision': recorded['revision'],
            'model_config_sha256': sha256(snapshot / 'config.json'),
            'data_protocol_sha256': sha256(PROCESSED / 'protocol.json')})
    print(json.dumps({'embedding_items': result.item_count, 'embeddings': str(result.embeddings_path),
                      'model_revision': recorded['revision']}), flush=True)


def tokenize(device):
    from cdr_framework.experiment_config import TokenizerTrainingConfig
    from experiments.train_tokenizer import run
    # AmazonPreprocessingConfig intentionally rejects Book/Movie. Supply explicit
    # labels to the already domain-general tokenizer; no Amazon schema mutation.
    run(TokenizerTrainingConfig.from_yaml(ASSET_CONFIG), device=device, force=False,
        source_domain='Book', target_domain='Movie')


def status():
    from experiments.run_llm_cdr import absolute_config
    config = absolute_config(MODEL_CONFIG)
    for name, path in (
        ('processed', PROCESSED / 'statistics.json'),
        ('embedding', ROOT / 'artifacts/embeddings/qwen3_8b/douban_book_movie/progress.json'),
        ('tokenizer_quality', ROOT / 'artifacts/tokenizer/qwen3_8b/douban_book_movie/quality_report.json'),
        ('training', config.output_dir / 'manifest.json'),
        ('validation_verification', TUNING / 'verification.json'),
        ('final_test', config.output_dir / 'test.json')):
        if not path.exists():
            print(f'{name}: pending', flush=True); continue
        data = json.loads(path.read_text(encoding='utf-8'))
        if name == 'training': data = {k: data.get(k) for k in ('complete', 'epochs_completed', 'best_validation')}
        if name == 'validation_verification': data = {k: data.get(k) for k in ('raw_scores_equal', 'rankings_equal', 'verified_all_metrics', 'metric_differences')}
        if name == 'final_test': data = {k: data.get(k) for k in ('state', 'epoch', 'metrics')}
        print(name + ': ' + json.dumps(data, ensure_ascii=False), flush=True)


def workflow(action, device):
    from experiments.recover_llm_evaluation import operation_lock
    from experiments.run_llm_cdr import absolute_config
    from cdr_framework.llm_training import config_payload
    if action == 'status': status(); return
    if action in ('embedding', 'tokenize'):
        verify_processed(PROCESSED, _identity(ARCHIVE_SHA256))
        with operation_lock(ROOT / f'artifacts/douban_book_movie/{action}_worker'):
            (embedding if action == 'embedding' else tokenize)(device)
        return
    with operation_lock(ROOT / 'artifacts/douban_book_movie/controller'):
        if action in ('convert', 'prepare', 'all'):
            download_archive(ARCHIVE)
            convert_archive(ARCHIVE, PROCESSED)
        else:
            verify_processed(PROCESSED, _identity(ARCHIVE_SHA256))
        if action == 'convert': return
        stages = ('prepare', 'run') if action == 'all' else (action,)
        for stage in stages:
            if stage == 'run':
                if config_payload(absolute_config(RUNTIME)) != config_payload(absolute_config(MODEL_CONFIG)):
                    raise RuntimeError('Douban runtime configuration differs; prepare matching config first')
            for command in commands(stage, device):
                print('$ ' + ' '.join(command), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
    print('=== DOUBAN WORKFLOW COMPLETED ===', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('convert', 'prepare', 'run', 'all', 'status', 'embedding', 'tokenize'), default='all')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    workflow(args.action, args.device)


if __name__ == '__main__':
    main()
