"""Separate model fitting; input assets are never written by this module."""
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cdr_framework.datasets.gencdr_protocol import sha
from cdr_framework.experiment_control import write_json_atomic
from cdr_framework.formal_training import _save_atomic
from . import MODEL_LABEL
from .assets import ExistingAssets, read
from .decoding import DomainTrie, beam_rank, ranking_metrics
from .model import GenCDRDecoder
from .protocol import REVERSE, bind, metric_key, verify_files
from .tokenizer import AdaptiveRQVAE


@dataclass(frozen=True)
class BaselineConfig:
    model: str = 'Qwen/Qwen3-0.6B'
    revision: str = 'c1899de289a04d12100db370d81485cdf75e47ca'
    hidden: tuple = (1024, 512)
    latent: int = 256
    levels: int = 3
    codebook_size: int = 2048
    tokenizer_rank: int = 64
    tokenizer_epochs: int = 100
    tokenizer_domain_epochs: int = 50
    tokenizer_batch_size: int = 512
    tokenizer_lr: float = 1e-4
    tokenizer_domain_lr: float = 5e-5
    universal_epochs: int = 10
    domain_epochs: int = 20
    experts: int = 4
    rank: int = 64
    alpha: int = 128
    batch_size: int = 8
    gradient_accumulation: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.
    vib: float = 1e-3
    patience: int = 5
    max_history: int = 50
    beam_size: int = 50

    def __post_init__(self):
        ints = ('latent', 'levels', 'codebook_size', 'tokenizer_rank', 'tokenizer_epochs',
            'tokenizer_domain_epochs', 'tokenizer_batch_size', 'universal_epochs', 'domain_epochs',
            'experts', 'rank', 'alpha', 'batch_size', 'gradient_accumulation', 'patience', 'max_history')
        if any(type(getattr(self, n)) is not int or getattr(self, n) < 1 for n in ints):
            raise ValueError('Training counts/dimensions must be positive integers')
        if self.beam_size < 20 or not self.hidden or any(h < 1 for h in self.hidden):
            raise ValueError('Beam must cover top20; hidden layers required')
        if any(not 0 < x < 1 for x in (self.tokenizer_lr, self.tokenizer_domain_lr, self.learning_rate)):
            raise ValueError('Invalid learning rate')
        if self.weight_decay < 0 or self.vib < 0:
            raise ValueError('Regularizers must be nonnegative')

    def payload(self):
        return json.loads(json.dumps(asdict(self)))


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def load_bound(path, identity):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['identity'] != identity:
        raise RuntimeError(f'Checkpoint identity changed: {path}')
    return saved


def tokenizer_for(assets, config):
    return AdaptiveRQVAE(assets.vectors.shape[1], tuple(config.hidden), config.latent,
        config.levels, config.codebook_size, config.tokenizer_rank)


def fit_tokenizer(assets, output, config, identity, device):
    path = output/'tokenizer.pt'
    if path.exists():
        saved = load_bound(path, identity)
        if saved.get('complete'):
            return saved['codes']
    else:
        saved = None
    model = tokenizer_for(assets, config)
    if saved and saved['phase'] == 'domain':
        model.set_phase('domain')
    model.to(device)
    if saved:
        model.load_state_dict(saved['model'])
    data = TensorDataset(assets.vectors[1:], assets.domains[1:])
    phases = (('universal', config.tokenizer_epochs, config.tokenizer_lr),
              ('domain', config.tokenizer_domain_epochs, config.tokenizer_domain_lr))
    for phase, epochs, lr in phases:
        if saved and saved['phase'] == 'domain' and phase == 'universal':
            continue
        model.set_phase(phase)
        model.to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0)
        start = 0
        if saved and saved['phase'] == phase:
            start = saved['epoch']
            optimizer.load_state_dict(saved['optimizer'])
            restore_rng(saved['rng'])
        for epoch in range(start+1, epochs+1):
            model.train()
            total = 0.
            loader = DataLoader(data, batch_size=config.tokenizer_batch_size, shuffle=True, num_workers=0)
            for x, d in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(x.to(device), d.to(device), phase, vib=config.vib)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite tokenizer loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach())*len(x)
            print(json.dumps(dict(stage='tokenizer', phase=phase, epoch=epoch, epochs=epochs,
                                  loss=total/len(data))), flush=True)
            _save_atomic(path, dict(identity=identity, phase=phase, epoch=epoch, model=model.state_dict(),
                optimizer=optimizer.state_dict(), rng=rng_state(), complete=False))
        saved = None
    codes = torch.zeros((len(assets.ids), config.levels), dtype=torch.long)
    model.eval()
    for start in range(1, len(codes), config.tokenizer_batch_size):
        end = start+config.tokenizer_batch_size
        codes[start:end] = model.codes(assets.vectors[start:end].to(device), assets.domains[start:end].to(device)).cpu()
    quality = dict(items=len(codes)-1,
        collision_rate=1-len(set(map(tuple, codes[1:].tolist())))/(len(codes)-1),
        codebook_utilization=[len(codes[1:, i].unique())/config.codebook_size for i in range(config.levels)])
    _save_atomic(path, dict(identity=identity, phase='domain', model=model.state_dict(), codes=codes,
                           complete=True, quality=quality))
    write_json_atomic(output/'tokenizer_quality.json', quality)
    del model, optimizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return codes


def backbone_for(config, device, factory=None):
    if factory:
        return factory().to(device)
    from transformers import AutoModel
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(config.model, revision=config.revision, local_files_only=True,
        trust_remote_code=False, torch_dtype=dtype, attn_implementation='sdpa')
    if getattr(model.config, '_commit_hash', None) != config.revision:
        raise RuntimeError('Pinned backbone revision mismatch')
    model.config.use_cache = False
    return model.to(device)


def decoder_for(config, device, factory=None):
    return GenCDRDecoder(backbone_for(config, device, factory), config.levels, config.codebook_size,
        config.experts, config.rank, config.alpha).to(device)


def encode_context(row, assets, codes, config):
    result = [1]
    for name in ('source_items', 'target_items'):
        history = row[name][-config.max_history:]
        domain = int(assets.domains[history[0]])
        result.append(2+domain)
        for item in history:
            result.extend(4+k*config.codebook_size+code for k, code in enumerate(codes[item].tolist()))
    # Explicit prediction-domain marker; no label token enters the context.
    result.append(2+int(assets.domains[row['target_items'][0]]))
    return result


def collate(rows, assets, codes, config, device):
    contexts = [torch.tensor(encode_context(r, assets, codes, config), dtype=torch.long) for r in rows]
    tokens = nn.utils.rnn.pad_sequence(contexts, batch_first=True).to(device)
    mask = tokens.ne(0).long()
    labels = codes[[r['positive_target_item'] for r in rows]].to(device)
    return tokens, mask, labels


@torch.no_grad()
def evaluate(model, rows, assets, codes, config, device):
    model.eval()
    trie = DomainTrie(codes, assets.targets)
    rankings = []
    # Beam batches share history; no teacher forcing, reranker or oracle candidates.
    for number, row in enumerate(rows, 1):
        tokens, mask, _ = collate([row], assets, codes, config, device)
        def next_codes(prefixes):
            return model.next_log_probs(tokens.expand(len(prefixes), -1), mask.expand(len(prefixes), -1), prefixes, 'domain')
        rankings.append(beam_rank(next_codes, trie, set(row['target_items']), beam=config.beam_size))
        if number % 100 == 0:
            print(f'Evaluation examples {number}/{len(rows)}', flush=True)
    result = ranking_metrics(rankings, [r['positive_target_item'] for r in rows])
    result['positives_in_seen'] = sum(r['positive_target_item'] in r['target_items'] for r in rows)
    return result


def train_job(root, pair, seed, output, config, identity, device, factory=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bind(output/'identity.json', identity)
    bind(output/'config.json', config.payload())
    verify_files(root, identity['assets'])
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if (manifest.get('identity') != identity or not manifest.get('complete')
                or manifest['checkpoint_sha256'] != sha(output/'best.pt')
                or manifest['tokenizer_sha256'] != sha(output/'tokenizer.pt')):
            raise RuntimeError('Completed training evidence changed')
        return manifest
    seed_all(seed)
    assets = ExistingAssets(root, pair)
    reverse = ExistingAssets(root, REVERSE[pair])
    codes = fit_tokenizer(assets, output, config, identity, device)
    # Reset so tokenization run/reuse consumes no recommendation RNG budget.
    seed_all(seed+10000)
    model = decoder_for(config, device, factory)
    train, validation = assets.rows('train'), assets.rows('validation')
    universal_rows = train+assets.reverse_training(reverse)
    last = output/'last.pt'
    saved = load_bound(last, identity) if last.exists() else None
    if saved:
        model.load_portable(saved['model'])
    history, best, stale = (saved['history'], saved['best'], saved['stale']) if saved else ([], None, 0)
    for phase, epochs, rows in (('universal', config.universal_epochs, universal_rows),
                                ('domain', config.domain_epochs, train)):
        if saved and saved['phase'] == 'domain' and phase == 'universal':
            continue
        model.set_phase(phase)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate, weight_decay=config.weight_decay)
        start = 0
        if saved and saved['phase'] == phase:
            start = saved['epoch']
            optimizer.load_state_dict(saved['optimizer'])
            restore_rng(saved['rng'])
        if phase == 'domain' and stale >= config.patience:
            break
        for epoch in range(start+1, epochs+1):
            model.train()
            loader = DataLoader(rows, batch_size=config.batch_size, shuffle=True, num_workers=0,
                                collate_fn=lambda batch: batch)
            optimizer.zero_grad(set_to_none=True)
            total, started = 0., time.monotonic()
            for step, batch in enumerate(loader, 1):
                tokens, mask, labels = collate(batch, assets, codes, config, device)
                loss = model.loss(tokens, mask, labels, phase, config.vib)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite recommendation loss')
                # Scale partial accumulation groups correctly.
                group = min(config.gradient_accumulation, len(loader)-((step-1)//config.gradient_accumulation)*config.gradient_accumulation)
                (loss/group).backward()
                if step % config.gradient_accumulation == 0 or step == len(loader):
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                total += float(loss.detach())*len(batch)
                if step % 40 == 0:
                    print(f'phase={phase} epoch={epoch} batch={step}/{len(loader)} loss={float(loss.detach()):.6f}', flush=True)
            record = dict(phase=phase, epoch=epoch, loss=total/len(rows), seconds=time.monotonic()-started)
            if phase == 'domain':
                metrics = evaluate(model, validation, assets, codes, config, device)
                record['validation'] = metrics
                if best is None or metric_key(metrics) > metric_key(best['metrics']):
                    best, stale = dict(epoch=epoch, metrics=metrics), 0
                    _save_atomic(output/'best.pt', dict(identity=identity, model=model.portable_state(),
                        phase=phase, epoch=epoch, validation=metrics, tokenizer_sha256=sha(output/'tokenizer.pt')))
                else:
                    stale += 1
            history.append(record)
            write_json_atomic(output/'history.json', history)
            _save_atomic(last, dict(identity=identity, model=model.portable_state(), optimizer=optimizer.state_dict(),
                phase=phase, epoch=epoch, history=history, best=best, stale=stale, rng=rng_state()))
            print(json.dumps(record), flush=True)
            if phase == 'domain' and stale >= config.patience:
                break
        saved = None
    verify_files(root, identity['assets'])
    manifest = dict(complete=True, identity=identity, model_label=MODEL_LABEL, best_validation=best['metrics'],
        best_epoch=best['epoch'], checkpoint_sha256=sha(output/'best.pt'), tokenizer_sha256=sha(output/'tokenizer.pt'),
        selection='mean(NDCG@5,NDCG@10), mean(HR@5,HR@10), NDCG@10; exact ties earlier epoch',
        paper_reproduction=False, history_sha256=sha(output/'history.json'))
    bind(manifest_path, manifest)
    return manifest


def evaluate_job(root, pair, seed, output, config, identity, device, freeze, factory=None):
    if not freeze:
        raise RuntimeError('Validation freeze required before test')
    verify_files(root, identity['assets'])
    output = Path(output)
    if freeze.get('identity') != identity or freeze.get('checkpoint_sha256') != sha(output/'best.pt'):
        raise RuntimeError('Frozen checkpoint binding changed')
    manifest = read(output/'manifest.json')
    if (not manifest.get('complete') or manifest.get('identity') != identity
            or manifest['checkpoint_sha256'] != sha(output/'best.pt')
            or manifest['tokenizer_sha256'] != sha(output/'tokenizer.pt')):
        raise RuntimeError('Completed training binding changed')
    if (output/'test.json').exists():
        report = read(output/'test.json')
        if report['freeze'] != freeze or report['identity'] != identity:
            raise RuntimeError('Existing test binding changed')
        return report
    verify_files(root, identity['assets'])
    seed_all(seed+10000)
    assets = ExistingAssets(root, pair)
    state = load_bound(output/'best.pt', identity)
    tokenizer = load_bound(output/'tokenizer.pt', identity)
    if not tokenizer.get('complete') or state['tokenizer_sha256'] != sha(output/'tokenizer.pt'):
        raise RuntimeError('Tokenizer evidence changed')
    model = decoder_for(config, device, factory)
    model.load_portable(state['model'])
    model.set_phase('domain')
    metrics = evaluate(model, assets.rows('test'), assets, tokenizer['codes'], config, device)
    verify_files(root, identity['assets'])
    report = dict(complete=True, split='test', model_label=MODEL_LABEL, paper_reproduction=False,
        identity=identity, freeze=freeze, metrics=metrics)
    bind(output/'test.json', report)
    return report
