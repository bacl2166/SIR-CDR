"""Disposable train-only throughput measurements; no recommendation checkpoint is saved."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdr_framework import llm_training
from cdr_framework.formal_training import _sha256, _write_json
from cdr_framework.llm_efficiency import DeviceLossMeter, training_batch_to_device
from cdr_framework.text_data import collate_text_rows, collate_training_rows, load_training_rows
from cdr_framework.text_training import load_rows
from experiments.run_llm_cdr import absolute_config, check_cached_model


def candidate_configs(config, batch_sizes, checkpointing):
    if checkpointing not in ('on', 'off', 'both'):
        raise ValueError('checkpointing must be on, off or both')
    effective = config.batch_size * config.gradient_accumulation_steps
    options = (False, True) if checkpointing == 'both' else (checkpointing == 'on',)
    candidates = []
    for size in dict.fromkeys(batch_sizes):
        if type(size) is not int or size < 1 or effective % size:
            raise ValueError(f'Each batch size must be a positive divisor of effective batch {effective}')
        for enabled in options:
            candidates.append(replace(config, batch_size=size,
                gradient_accumulation_steps=effective // size, llm_gradient_checkpointing=enabled))
    return candidates


def sample_indices(row_count, count, seed):
    if row_count < 1 or count < 1:
        raise ValueError('Need positive row count and sample count')
    return torch.randint(row_count, (count,), generator=torch.Generator().manual_seed(seed)).tolist()


def require_benchmark_card(cards, gpu_id, own_pid=None):
    """Never benchmark on another process' GPU, even if much memory is free."""
    card = next((c for c in cards if c['index'] == gpu_id), None)
    if (card is None or set(card['pids']) - ({own_pid} if own_pid is not None else set())
            or card.get('memory_free_mib', 0) < 4096
            or (not card['pids'] and card.get('utilization', 100) > 5)):
        raise RuntimeError('Benchmark GPU absent or occupied by another process')
    return card


def benchmark_once(config, device, *, warmup_updates=1, measure_updates=3,
                   input_pipeline='legacy'):
    """Run complete optimizer updates on disposable weights using training rows only.

    Uniform seeded training rows measure compute, not model quality. Equal
    effective batches use the same row order. Batch-dependent CPF
    statistics are intentionally not claimed to be mathematically equivalent.
    """
    if warmup_updates < 1 or measure_updates < 1:
        raise ValueError('Warmup and measurement updates must be positive')
    if input_pipeline not in ('legacy', 'optimized'):
        raise ValueError('Unknown input pipeline')
    device = torch.device(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    identity = llm_training.signature(config)
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed)
        model, catalog = llm_training.build_model(config, device)
        rows = (load_rows(config, catalog, 'train') if input_pipeline == 'legacy' else
                load_training_rows(config, len(catalog.item_latents), catalog.target_item_ids))
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        model.train()
        meter = DeviceLossMeter()
        effective = config.batch_size * config.gradient_accumulation_steps
        indices = sample_indices(len(rows), (warmup_updates + measure_updates) * effective, config.seed)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = None
        for update in range(warmup_updates + measure_updates):
            if update == warmup_updates:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, effective, config.batch_size):
                sample = [rows[indices[update * effective + offset + i]]
                          for i in range(config.batch_size)]
                if input_pipeline == 'legacy':
                    batch, _ = collate_text_rows(sample)
                    batch = batch.to(device)
                else:
                    batch = training_batch_to_device(collate_training_rows(sample), device)
                losses = model(batch)
                if not torch.isfinite(losses['total']):
                    raise RuntimeError('Nonfinite benchmark loss')
                (losses['total'] * len(sample)).backward()
                if update >= warmup_updates:
                    meter.add(losses, len(sample))
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(effective)
            torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        return {
            'status': 'ok', 'training_identity': identity,
            'input_pipeline': input_pipeline,
            'sampling_seed': config.seed, 'sample_indices_sha256': hashlib.sha256(
                json.dumps(indices).encode()).hexdigest(),
            'batch_size': config.batch_size,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'llm_gradient_checkpointing': config.llm_gradient_checkpointing,
            'effective_batch_size': effective, 'warmup_updates': warmup_updates,
            'optimizer_updates_measured': measure_updates,
            'measured_examples': meter.examples, 'seconds': seconds,
            'train_samples_per_second': meter.examples / seconds,
            'peak_cuda_GiB': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
            'mean_losses': meter.means(), 'checkpoint_written': False,
            'data_policy': 'uniform seeded train-only rows with replacement; no validation/test scoring',
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16])
    parser.add_argument('--checkpointing', choices=('on', 'off', 'both'), default='off')
    parser.add_argument('--warmup-updates', type=int, default=1)
    parser.add_argument('--measure-updates', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu-id', type=int, help='Physical idle GPU index; required for CUDA')
    parser.add_argument('--input-pipeline', choices=('legacy', 'optimized'), default='legacy')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Benchmark report already exists; choose another output')
    if min(args.warmup_updates, args.measure_updates) < 1:
        parser.error('Warmup and measurement updates must be positive')
    config = absolute_config(args.config)
    device = torch.device(args.device)
    inventory = None
    if device.type == 'cuda':
        if args.gpu_id is None or args.gpu_id < 0:
            parser.error('CUDA benchmark requires an explicit --gpu-id')
        from cdr_framework.experiment_control import query_gpu_inventory
        inventory = query_gpu_inventory
        card = require_benchmark_card(inventory(), args.gpu_id)
        inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
        if inherited not in (None, '', card['uuid']):
            parser.error('Unset inherited CUDA_VISIBLE_DEVICES or use the approved full UUID')
        os.environ['CUDA_VISIBLE_DEVICES'] = card['uuid']
        device = torch.device('cuda:0')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    check_cached_model(config)
    configs = candidate_configs(config, args.batch_sizes, args.checkpointing)
    environment = {'device': str(device), 'torch': torch.__version__, 'cpu_threads': torch.get_num_threads(),
                   'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    report = {'schema': 'llm-throughput-v2', 'benchmark_sha256': _sha256(Path(__file__)),
              'environment': environment, 'results': [],
              'note': 'Performance probe only. No validation selection, final metrics or training checkpoint.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for current in configs:
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            before = require_benchmark_card(inventory(), args.gpu_id, own_pid=os.getpid())
        else:
            before = None
        print(f'BENCHMARK batch={current.batch_size} accumulation={current.gradient_accumulation_steps} '
              f'checkpointing={current.llm_gradient_checkpointing}', flush=True)
        try:
            result = benchmark_once(current, device, warmup_updates=args.warmup_updates,
                                    measure_updates=args.measure_updates, input_pipeline=args.input_pipeline)
        except torch.cuda.OutOfMemoryError:
            result = {'status': 'cuda_oom', 'batch_size': current.batch_size,
                      'llm_gradient_checkpointing': current.llm_gradient_checkpointing}
        if inventory is not None:
            after = next(c for c in inventory() if c['index'] == args.gpu_id)
            result['gpu_before'], result['gpu_after'] = before, after
            if set(after['pids']) - {os.getpid()}:
                result['status'] = 'resource_conflict'
        report['results'].append(result)
        _write_json(args.output, report)
        print(json.dumps({k: v for k, v in result.items() if k not in ('training_identity', 'mean_losses')}), flush=True)
    print(f'Performance report: {args.output}', flush=True)
    if not any(result['status'] == 'ok' for result in report['results']):
        raise RuntimeError('No benchmark candidate succeeded')


if __name__ == '__main__':
    main()
