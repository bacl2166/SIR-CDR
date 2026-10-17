"""Small validation-only probe for cached/direct score drift. Never trains or tests."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdr_framework import llm_training
from cdr_framework.formal_training import _sha256, _write_json
from cdr_framework.text_data import collate_text_rows
from cdr_framework.text_training import load_rows
from experiments.run_llm_cdr import absolute_config
from experiments.tune_llm_fusion import collect_scores, validate_checkpoint


def original_batches(length, batch_size, changed, cases=3):
    """Include complete original batches so history padding/batch shape is unchanged."""
    if min(length, batch_size, cases) < 1 or not changed:
        raise ValueError('Need nonempty rows and positive batch size/cases')
    if any(type(i) is not int or i < 0 or i >= length for i in changed):
        raise ValueError('Audit row is outside validation split')
    starts = sorted({i // batch_size * batch_size for i in changed[:cases]})
    return [i for start in starts for i in range(start, min(start + batch_size, length))]


def _tensors(value, prefix=''):
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if hasattr(value, 'last_hidden_state'):
        return _tensors(value.last_hidden_state, prefix + '.last_hidden_state')
    if isinstance(value, (dict, tuple, list)):
        result = {}
        iterator = value.items() if isinstance(value, dict) else enumerate(value)
        for key, entry in iterator:
            result.update(_tensors(entry, f'{prefix}.{key}'))
        return result
    return {}


def _difference(before, after):
    a, b = _tensors(before), _tensors(after)
    result = {}
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b or a[name].shape != b[name].shape:
            result[name] = {'equal': False, 'shape_mismatch': True}
            continue
        equal = torch.equal(a[name], b[name])
        result[name] = {'equal': equal,
                        'max_abs': 0. if equal else float((a[name].float() - b[name].float()).abs().max())}
    return {'all_equal': set(a) == set(b) and all(v['equal'] for v in result.values()), 'tensors': result}


def compare_scores(before, after):
    """Align by item IDs: top-k candidate order can change between scoring passes."""
    order_equal = torch.equal(before['candidates'], after['candidates'])
    deltas = {'retrieval': 0., 'generation': 0.}
    different_sets = nonfinite = 0
    for row in range(len(before['candidates'])):
        a = {int(item): pos for pos, item in enumerate(before['candidates'][row])}
        b = {int(item): pos for pos, item in enumerate(after['candidates'][row])}
        different_sets += set(a) != set(b)
        for item in a.keys() & b.keys():
            for key in deltas:
                x, y = float(before[key][row, a[item]]), float(after[key][row, b[item]])
                if x == y:
                    continue
                if not math.isfinite(x) or not math.isfinite(y):
                    nonfinite += 1
                else:
                    deltas[key] = max(deltas[key], abs(x - y))
    return {'bitwise_equal': all(torch.equal(before[k], after[k]) for k in ('candidates', 'retrieval', 'generation')),
            'candidate_order_equal': order_equal, 'different_candidate_set_rows': different_sets,
            'nonfinite_mask_mismatches': nonfinite, 'maximum_common_item_score_difference': deltas}


@torch.no_grad()
def _traced_encode(model, batch, items):
    records, handles, counters = {}, [], {}
    def hook(name):
        def capture(module, inputs, output):
            index = counters.get(name, 0)
            counters[name] = index + 1
            for key, tensor in _tensors(output).items():
                records[f'{name}[{index}]{key}'] = tensor.detach().cpu().clone()
        return capture
    for name in ('source_gru', 'target_gru', 'reasoner.project', 'reasoner.backbone', 'reasoner.memory'):
        handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
    try:
        state = model.encode(batch, items)
        for name in ('query', 'prefix', 'states'):
            records[f'final.{name}'] = state[name].cpu().clone()
        return records
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def probe_model(model, rows, config, device, cached):
    model.eval()
    first, second = model.item_states(), model.item_states()
    catalog = _difference(first, second)
    batch, _ = collate_text_rows(rows[:config.evaluation_batch_size])
    batch = batch.to(device)
    trace1 = _traced_encode(model, batch, first)
    trace2 = _traced_encode(model, batch, first)
    trace3 = _traced_encode(model, batch, second)
    del first, second
    a = collect_scores(model, rows, config, device, 1000)
    b = collect_scores(model, rows, config, device, 1000)
    return {'probe_examples': len(rows), 'training_modules': [n for n, m in model.named_modules() if m.training],
            'repeated_catalog': catalog,
            'fixed_catalog_encoding': _difference(trace1, trace2),
            'rebuilt_catalog_encoding': _difference(trace1, trace3),
            'repeated_scores': compare_scores(a, b),
            'old_cache_vs_probe': compare_scores(cached, a)}


def _child(args):
    config = absolute_config(args.config)
    checkpoint = config.output_dir / 'best_model.pt'
    state, identity = validate_checkpoint(config, checkpoint)
    verification = json.loads((args.parameters / 'verification.json').read_text(encoding='utf-8'))
    stored = torch.load(args.cache_root / 'validation_scores.pt', map_location='cpu', weights_only=True)
    if not stored['complete'] or stored['identity']['checkpoint_sha256'] != _sha256(checkpoint):
        raise RuntimeError('Cache/checkpoint identity mismatch')
    if stored['identity']['training_identity'] != identity:
        raise RuntimeError('Cache training identity differs')
    if verification['identity']['cache_sha256'] != _sha256(args.cache_root / 'validation_scores.pt'):
        raise RuntimeError('Verification refers to another cache')
    device = torch.device(args.device)
    report = {'profile': args.profile, 'training_identity': identity,
              'checkpoint_sha256': _sha256(checkpoint), 'diagnostic_sha256': _sha256(__file__),
              'policy': {'dtype': config.llm_dtype if args.profile == 'native' else 'float32',
                         'deterministic_algorithms': args.profile == 'controlled',
                         'math_sdpa_only': args.profile == 'controlled',
                         'cudnn_disabled': args.profile == 'controlled'},
              'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                              'workspace': os.getenv('CUBLAS_WORKSPACE_CONFIG'), 'device': str(device)}}
    try:
        if args.profile == 'controlled':
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        report['environment']['actual_flags'] = {
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'nvidia_tf32_override': os.getenv('NVIDIA_TF32_OVERRIDE'),
        }
        torch.manual_seed(config.seed)
        model, catalog = llm_training.build_model(config, device)
        llm_training.restore_state(model, state['model'])
        rows = load_rows(config, catalog, 'validation')
        changed = [r['row'] for r in verification['audit']['positive_rank_changes']]
        indices = original_batches(len(rows), config.evaluation_batch_size, changed, args.cases)
        report['original_validation_rows'] = indices
        old = {key: value[indices] for key, value in stored['scores'].items()}
        if args.profile == 'controlled':
            from torch.nn.attention import sdpa_kernel, SDPBackend
            model.float()  # Disposable inference policy, never saved over the checkpoint.
            context = sdpa_kernel(SDPBackend.MATH)
        else:
            context = nullcontext()
        with context:
            report['probe'] = probe_model(model, [rows[i] for i in indices], config, device, old)
        report['status'] = 'complete'
    except RuntimeError as error:
        # Deterministic mode may expose an unsupported CUDA operation. Preserve
        # the actual error instead of falling back silently or relaxing checks.
        report.update(status='failed', error=repr(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, report)
    summary = {'profile': args.profile, 'status': report['status'], 'report': str(args.output)}
    if 'probe' in report:
        probe = report['probe']
        summary.update(catalog_equal=probe['repeated_catalog']['all_equal'],
                       encoding_equal=probe['fixed_catalog_encoding']['all_equal'],
                       repeated_scores=probe['repeated_scores'])
    else:
        summary['error'] = report['error']
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', choices=('phones_to_electronics', 'sports_to_clothing'), default='phones_to_electronics')
    parser.add_argument('--config')
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile', choices=('both', 'native', 'controlled'), default='both')
    parser.add_argument('--cases', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    tag = 'qwen3_0_6b_fast_b4_eb4_gc0'
    args.config = args.config or str(ROOT / f'configs/runtime/{args.pair}_{tag}.yaml')
    args.cache_root = args.cache_root or ROOT / f'artifacts/tuning/{args.pair}_{tag}_seed42'
    args.parameters = args.parameters or Path(str(args.cache_root) + '_parameters')
    if args.cases < 1:
        parser.error('--cases must be positive')
    if args.profile != 'both':
        if args.output is None:
            parser.error('Internal single-profile invocation requires --output')
        _child(args)
        return
    output = args.output or ROOT / f'logs/scoring_diagnostic_{args.pair}.json'
    if output.exists():
        parser.error('Diagnostic report already exists; choose a new --output')
    profiles = []
    for profile in ('native', 'controlled'):
        child_output = output.with_name(output.stem + f'_{profile}.json')
        if child_output.exists():
            parser.error(f'Diagnostic profile already exists: {child_output}')
        environment = os.environ.copy()
        if profile == 'controlled':
            environment.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', NVIDIA_TF32_OVERRIDE='0')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--profile', profile,
                        '--config', str(args.config), '--cache-root', str(args.cache_root),
                        '--parameters', str(args.parameters), '--output', str(child_output),
                        '--cases', str(args.cases), '--device', args.device], cwd=ROOT, env=environment, check=True)
        profiles.append(json.loads(child_output.read_text(encoding='utf-8')))
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, {'diagnostic_only': True, 'profiles': profiles,
                         'note': 'Controlled profile is a probe, not a verified repair or a final recommendation metric.'})
    print(f'Diagnostic complete: {output}', flush=True)


if __name__ == '__main__':
    main()
