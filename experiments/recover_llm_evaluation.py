"""Resume evaluation with a signed CPU catalog snapshot; never retrain or bypass identity checks."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdr_framework import llm_training
from cdr_framework.formal_training import _sha256, _save_atomic, _write_json, load_fixed_catalog
from cdr_framework.text_training import load_rows
from experiments.run_llm_cdr import absolute_config
from experiments.tune_llm_fusion import validate_checkpoint, score_metrics
from experiments.tune_llm_parameters import parameter_settings, select_protected
from experiments.llm_snapshot_scoring import cpu_item_snapshot, collect_snapshot_scores, rank_snapshot_parameters


POLICY = {'schema': 'llm-snapshot-evaluation-v1', 'catalog': 'cpu-float32-single-thread-checkpoint-snapshot',
          'execution_dtype': 'float32', 'sdpa': 'math', 'cudnn': False, 'tf32': False,
          'deterministic_algorithms': True, 'tie_policy': 'score-descending-item-id-ascending',
          'maximum_candidates': 1000, 'score_rounding': False}


@contextmanager
def operation_lock(output):
    """OS releases this lock on process exit or server shutdown; no stale PID deletion."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'operation.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('Another recovery worker holds this output') from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError('Another recovery worker holds this output') from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def fixed_execution(device, seed):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if device.type == 'cuda' and os.getenv('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError('Use the recovery CLI to set cuBLAS workspace before CUDA initialization')
    previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=devices), sdpa_kernel(SDPBackend.MATH):
            torch.manual_seed(seed)
            yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        (torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
         torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32) = previous[2:]


def _identity(config, checkpoint, training_identity, device):
    files = ('recover_llm_evaluation.py', 'llm_snapshot_scoring.py', 'tune_llm_fusion.py', 'tune_llm_parameters.py')
    runtime = {'device': str(device), 'cuda': torch.version.cuda}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        runtime.update(gpu=properties.name, capability=[properties.major, properties.minor],
                       memory=properties.total_memory, uuid=str(getattr(properties, 'uuid', 'unavailable')))
    return json.loads(json.dumps({'training_identity': training_identity,
        'checkpoint_sha256': _sha256(checkpoint), 'source_config': llm_training.config_payload(config),
        'completed_manifest_sha256': _sha256(config.output_dir / 'manifest.json'),
        'evaluation_policy': POLICY, 'runtime': runtime,
        'code': {name: _sha256(ROOT / 'experiments' / name) for name in files}}))


def _load_bound(path, identity):
    stored = torch.load(path, map_location='cpu', weights_only=True)
    if stored.get('identity') != identity:
        raise RuntimeError(f'Evaluation identity changed: {path}; use an independent output')
    return stored


def _model(config, state, device):
    model, catalog = llm_training.build_model(config, device)
    llm_training.restore_state(model, state['model'])
    model.float().eval()  # Inference execution policy only; never saved over training weights.
    return model, catalog


def recover_stage(config, output, stage, device):
    output, device = Path(output), torch.device(device)
    if stage not in ('cache', 'verify', 'test'):
        raise ValueError('Unknown recovery stage')
    checkpoint = config.output_dir / 'best_model.pt'
    state, training_identity = validate_checkpoint(config, checkpoint)
    manifest_path = config.output_dir / 'manifest.json'
    if not manifest_path.exists():
        raise RuntimeError('Completed training manifest is required before evaluation recovery')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('complete') is not True or manifest.get('identity') != state['identity']
            or manifest.get('best_validation') != state.get('validation')
            or not 0 < state['epoch'] <= manifest.get('epochs_completed', 0)):
        raise RuntimeError('Completed training manifest/checkpoint mismatch')
    if config.retrieval_score_weight != 1.:
        raise ValueError('This fixed finite search requires retrieval_score_weight=1')
    test_path = config.output_dir / 'test.json'
    if stage != 'test' and test_path.exists():
        raise RuntimeError('Existing test report/reservation: validation recovery refused')
    with operation_lock(output), fixed_execution(device, config.seed):
        identity = _identity(config, checkpoint, training_identity, device)
        snapshot_path = output / 'item_states.pt'
        cache_path = output / 'validation_scores.pt'
        search_path = output / 'search.json'
        verify_path = output / 'verification.json'
        if stage == 'cache':
            model = None
            if snapshot_path.exists():
                snapshot = _load_bound(snapshot_path, identity)['items']
            else:
                model, _ = _model(config, state, device)
                snapshot = cpu_item_snapshot(model)
                _save_atomic(snapshot_path, {'identity': identity, 'items': snapshot})
            snapshot_hash = _sha256(snapshot_path)
            cached = None
            complete = False
            if cache_path.exists():
                stored = _load_bound(cache_path, identity)
                if stored['snapshot_sha256'] != snapshot_hash:
                    raise RuntimeError('Catalog snapshot changed')
                cached, complete = stored['scores'], stored['complete']
            if not complete:
                if model is None:
                    model, _ = _model(config, state, device)
                catalog = load_fixed_catalog(config)
                rows = load_rows(config, catalog, 'validation')
                if cached is not None and cached['positive'].tolist() != [r.positive_target_item for r in rows[:len(cached['positive'])]]:
                    raise RuntimeError('Partial validation cache row order differs')
                def progress(scores, count):
                    _save_atomic(cache_path, {'identity': identity, 'snapshot_sha256': snapshot_hash,
                        'scores': scores, 'complete': count == len(rows)})
                    print(f'Snapshot validation cache: {count}/{len(rows)}', flush=True)
                cached = collect_snapshot_scores(model, rows, config, device, snapshot, cached, progress)
                progress(cached, len(rows))
            cache_hash = _sha256(cache_path)
            if search_path.exists():
                report = json.loads(search_path.read_text(encoding='utf-8'))
                if (report['identity'] != identity or report['cache_sha256'] != cache_hash
                        or report['snapshot_sha256'] != snapshot_hash):
                    raise RuntimeError('Completed selection differs; refusing overwrite')
                print('Reusing completed snapshot validation and selection', flush=True)
                return report
            results = []
            for setting in parameter_settings(config):
                ranked = rank_snapshot_parameters(cached, setting, max(config.top_ks))
                results.append({'setting': setting, 'metrics': score_metrics(cached, ranked, config.top_ks)})
            best = select_protected(results)
            report = {'identity': identity, 'snapshot_sha256': snapshot_hash, 'cache_sha256': cache_hash,
                      'complete': True, 'split': 'validation', 'results': results, 'best': best,
                      'selection_policy': 'all four >= runtime anchor, then N@10/R@10/N@5/R@5',
                      'candidate_evaluations_this_policy': len(results),
                      'prior_native_search_is_preserved_and_not_reused': True}
            _write_json(search_path, report)
            print(json.dumps({'selected_setting': best['setting'], 'validation_metrics': best['metrics']}), flush=True)
            return report

        snapshot = _load_bound(snapshot_path, identity)['items']
        stored = _load_bound(cache_path, identity)
        search = json.loads(search_path.read_text(encoding='utf-8'))
        if (search['identity'] != identity or not stored['complete'] or not search['complete']
                or stored['snapshot_sha256'] != _sha256(snapshot_path)
                or search['snapshot_sha256'] != _sha256(snapshot_path)
                or search['cache_sha256'] != _sha256(cache_path)):
            raise RuntimeError('Snapshot/cache/selection identity mismatch')
        binding = {'identity': identity, 'snapshot_sha256': _sha256(snapshot_path),
                   'cache_sha256': _sha256(cache_path), 'search_sha256': _sha256(search_path)}
        selected = search['best']['setting']
        if stage == 'verify':
            if verify_path.exists():
                old = json.loads(verify_path.read_text(encoding='utf-8'))
                if old.get('binding') == binding and old.get('raw_scores_equal') and old.get('rankings_equal'):
                    print('Reusing matching exact verification', flush=True)
                    return old
            model, catalog = _model(config, state, device)
            rows = load_rows(config, catalog, 'validation')
            fresh = collect_snapshot_scores(model, rows, config, device, snapshot,
                progress=lambda _, n: print(f'Snapshot verify: {n}/{len(rows)}', flush=True))
            cached = stored['scores']
            raw_equal = all(cached[k].dtype == fresh[k].dtype and torch.equal(cached[k], fresh[k]) for k in cached)
            before = rank_snapshot_parameters(cached, selected, max(config.top_ks))
            after = rank_snapshot_parameters(fresh, selected, max(config.top_ks))
            ranks_equal = torch.equal(before, after)
            fresh_metrics = score_metrics(fresh, after, config.top_ks)
            differences = {k: fresh_metrics[k] - value
                           for k, value in score_metrics(cached, before, config.top_ks).items()
                           if k.startswith(('R@', 'N@'))}
            result = {'binding': binding, 'raw_scores_equal': raw_equal, 'rankings_equal': ranks_equal,
                      'metric_differences': differences, 'verified_all_metrics': all(v == 0 for v in differences.values())}
            _write_json(verify_path, result)
            print(json.dumps(result | {'binding': 'saved in verification.json'}), flush=True)
            if not raw_equal or not ranks_equal or not result['verified_all_metrics']:
                raise RuntimeError('Snapshot evaluation scores/ranks differ; test remains blocked')
            return result

        verification = json.loads(verify_path.read_text(encoding='utf-8'))
        if (verification.get('binding') != binding or not verification.get('raw_scores_equal')
                or not verification.get('rankings_equal') or not verification.get('verified_all_metrics')):
            raise RuntimeError('Exact validation verification has not passed')
        test_identity = binding | {'verification_sha256': _sha256(verify_path), 'split': 'test'}
        if test_path.exists():
            saved = json.loads(test_path.read_text(encoding='utf-8'))
            if saved.get('state') == 'complete' and saved.get('recovery_identity') == test_identity:
                print(json.dumps({'state': 'complete', 'reused': True,
                                  'test_report': str(test_path), 'metrics': saved['metrics']}), flush=True)
                return saved
            raise RuntimeError('Existing test result/reservation differs or is unfinished; refusing overwrite')
        reservation = {'state': 'reserved', 'pid': os.getpid(), 'recovery_identity': test_identity}
        with test_path.open('x', encoding='utf-8') as stream:
            json.dump(reservation, stream)
        try:
            with (config.output_dir / 'test.lock').open('x', encoding='utf-8') as stream:
                json.dump(reservation, stream)
            model, catalog = _model(config, state, device)
            rows = load_rows(config, catalog, 'test')
            started = time.perf_counter()
            scores = collect_snapshot_scores(model, rows, config, device, snapshot,
                progress=lambda _, n: print(f'Snapshot test: {n}/{len(rows)}', flush=True))
            ranked = rank_snapshot_parameters(scores, selected, max(config.top_ks))
            metrics = score_metrics(scores, ranked, config.top_ks)
            metrics.update(seconds=time.perf_counter() - started,
                coverage=len(set(ranked.flatten().tolist()) - {0}) / len(catalog.target_item_ids),
                positives_in_seen=sum(r.positive_target_item in r.seen_items for r in rows))
            if json.loads(test_path.read_text(encoding='utf-8')) != reservation:
                raise RuntimeError('Test reservation changed')
            result = {'state': 'complete', 'split': 'test', 'epoch': state['epoch'],
                      'checkpoint_sha256': _sha256(checkpoint), 'evaluation_policy': POLICY,
                      'selected_setting': selected, 'recovery_identity': test_identity, 'metrics': metrics}
            _write_json(test_path, result)
            _write_json(config.output_dir / 'test.lock', {'state': 'complete', 'recovery_identity': test_identity})
            print(json.dumps({'state': 'complete', 'test_report': str(test_path),
                              'selected_setting': selected, 'metrics': metrics}), flush=True)
            return result
        except BaseException as error:
            if json.loads(test_path.read_text(encoding='utf-8')) == reservation:
                _write_json(test_path, reservation | {'state': 'failed', 'error': repr(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', choices=('phones_to_electronics', 'sports_to_clothing'), default='phones_to_electronics')
    parser.add_argument('--config')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--action', choices=('all', 'cache', 'verify', 'test'), default='all')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    tag = 'qwen3_0_6b_fast_b4_eb4_gc0'
    path = Path(args.config) if args.config else ROOT / f'configs/runtime/{args.pair}_{tag}.yaml'
    config = absolute_config(path)
    tag = config.output_dir.parent.parent.name
    output = args.output or ROOT / f'artifacts/tuning/{args.pair}_{tag}_seed{config.seed}_snapshot_eval_v1'
    if args.action == 'all':
        stages = ('test',) if (config.output_dir / 'test.json').exists() else ('cache', 'verify', 'test')
        environment = os.environ.copy()
        environment.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', NVIDIA_TF32_OVERRIDE='0')
        for stage in stages:
            print(f'=== SNAPSHOT {stage.upper()} ===', flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--action', stage,
                            '--config', str(path.resolve()), '--output', str(output.resolve()),
                            '--device', args.device], cwd=ROOT, env=environment, check=True)
        print('=== SNAPSHOT EVALUATION COMPLETED ===', flush=True)
    else:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
        recover_stage(config, output, args.action, torch.device(args.device))


if __name__ == '__main__':
    main()
