"""Serial 0.6B efficiency experiments: prepare, train, validation tune, verify, test."""
import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdr_framework.llm_config import LLMCDRConfig
from cdr_framework.llm_training import config_payload
from experiments.run_llm_cdr import absolute_config, prepare_model, preflight, smoke


TASKS = {'phones_to_electronics': 'phones_electronics', 'sports_to_clothing': 'sports_clothing'}


def runtime_config(pair, root=ROOT, *, batch_size=4, evaluation_batch_size=4, checkpointing=False):
    if pair not in TASKS:
        raise ValueError('Only the two prepared Amazon tasks are enabled; Douban assets are pending')
    if type(batch_size) is not int or batch_size < 1 or 16 % batch_size:
        raise ValueError('batch_size must be a positive divisor of nominal effective batch 16')
    if type(evaluation_batch_size) is not int or evaluation_batch_size < 1:
        raise ValueError('evaluation_batch_size must be positive')
    base = LLMCDRConfig.from_yaml(ROOT / f'configs/llm_{TASKS[pair]}_qwen06_fast.yaml')
    root = Path(root).resolve()
    tag = f'qwen3_0_6b_fast_b{batch_size}_eb{evaluation_batch_size}_gc{int(checkpointing)}'
    return replace(base, batch_size=batch_size, gradient_accumulation_steps=16 // batch_size,
                   evaluation_batch_size=evaluation_batch_size, llm_gradient_checkpointing=checkpointing,
                   processed_dir=root / base.processed_dir, tokenizer_dir=root / base.tokenizer_dir,
                   output_dir=root / 'artifacts/llm_cdr_v1' / tag / pair / f'seed{base.seed}')


def _run(command):
    print('$ ' + ' '.join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_pair(config, path, tuning, *, device='cuda'):
    """Reuse completed training; exact snapshot verification gates the final test."""
    python = sys.executable
    cli = [python, str(ROOT / 'experiments/run_llm_cdr.py'), '--config', str(path)]
    print(f'=== TRAIN {config.output_dir} ===', flush=True)
    _run(cli + ['--action', 'train', '--device', device])
    print('=== SNAPSHOT VALIDATION AND FINAL TEST ===', flush=True)
    action = 'test' if (config.output_dir / 'test.json').exists() else 'all'
    _run([python, str(ROOT / 'experiments/recover_llm_evaluation.py'), '--config', str(path),
          '--output', str(tuning) + '_snapshot_eval_v1', '--action', action, '--device', device])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'benchmark', 'run'), required=True)
    parser.add_argument('--pairs', nargs='+', choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--evaluation-batch-size', type=int, default=4)
    parser.add_argument('--checkpointing', choices=('on', 'off'), default='off')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    for pair in args.pairs:
        config = runtime_config(pair, batch_size=args.batch_size,
                                evaluation_batch_size=args.evaluation_batch_size,
                                checkpointing=args.checkpointing == 'on')
        tag = config.output_dir.parent.parent.name
        path = ROOT / 'configs/runtime' / f'{pair}_{tag}.yaml'
        tuning = ROOT / 'artifacts/tuning' / f'{pair}_{tag}_seed{config.seed}'
        if args.action == 'prepare':
            print(f'=== PREPARE {pair} ===', flush=True)
            prepared = prepare_model(config, path)
            preflight(prepared)
            smoke(prepared, torch.device(args.device))
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
            continue
        if not path.is_file():
            raise FileNotFoundError(f'Run --action prepare with matching batch/checkpoint settings first: {path}')
        if config_payload(absolute_config(path)) != config_payload(config):
            raise RuntimeError('Runtime config changed; do not mix recipes or output identities')
        if args.action == 'benchmark':
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            _run([sys.executable, str(ROOT / 'experiments/benchmark_llm_cdr.py'),
                  '--config', str(path), '--device', args.device, '--checkpointing', 'both',
                  '--output', str(ROOT / 'logs' / f'benchmark_{pair}_{stamp}.json')])
        else:
            run_pair(config, path, tuning, device=args.device)
    print(f'=== {args.action.upper()} COMPLETED ===', flush=True)


if __name__ == '__main__':
    main()
