"""RQ-VAE from the public GenCDR algorithm; missing routers reconstructed.

Reference: hupeiyu21/GenCDR@204b0ce, quantization/rqvae/{layers,rqvae}.py.
The public code uses EMA codebooks and cumulative commitment losses. This
implementation preserves those computations, fixes singleton residual shapes,
and adds the paper's domain LoRA and variational routing (not published upstream).
No interaction labels enter tokenizer fitting.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        widths = (input_dim, *hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU())
                                    for a, b in zip(widths, widths[1:])])
        self.residuals = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in zip(widths, widths[1:])])
        self.output = nn.Linear(hidden[-1], output_dim)

    def forward(self, x):
        for block, residual in zip(self.blocks, self.residuals):
            x = block(x) + residual(x)
        return self.output(x)


class LowRankLinear(nn.Module):
    def __init__(self, base, rank, alpha=None):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = (alpha or rank) / rank
        self.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


def adapted_encoder(encoder, rank):
    result = copy.deepcopy(encoder)
    result.requires_grad_(False)
    for name, child in list(result.named_children()):
        if isinstance(child, nn.Linear):
            setattr(result, name, LowRankLinear(child, rank))
        else:
            replace_linear(child, rank)
    return result


def replace_linear(module, rank):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LowRankLinear(child, rank))
        else:
            replace_linear(child, rank)


class VariationalRouter(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.stats = nn.Linear(hidden, 2)

    def forward(self, x):
        mu, logvar = self.stats(self.hidden(x)).chunk(2, -1)
        logvar = logvar.clamp(-10, 5)
        z = mu + torch.randn_like(mu) * (logvar*.5).exp() if self.training else mu
        kl = .5*(mu.square()+logvar.exp()-1-logvar).mean()
        return z.sigmoid(), kl


class ResidualEMA(nn.Module):
    def __init__(self, levels, size, latent, decay=.99):
        super().__init__()
        self.levels, self.size, self.decay = levels, size, decay
        self.register_buffer('books', torch.randn(levels, size, latent))
        self.register_buffer('counts', torch.zeros(levels, size))
        self.register_buffer('sums', self.books.clone())
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, z):
        residual, aggregate, losses, codes = z.detach().clone(), torch.zeros_like(z), [], []
        for level in range(self.levels):
            if self.training and not bool(self.initialized):
                repeat = residual.repeat((self.size+len(z)-1)//len(z), 1)
                self.books[level].copy_(repeat[:self.size] + torch.rand_like(repeat[:self.size])*.01/math.sqrt(z.shape[-1]))
                self.sums[level].copy_(self.books[level])
                self.counts[level].fill_(1)
            book = self.books[level]
            distances = residual.square().sum(1, keepdim=True) + book.square().sum(1)[None] - 2*residual@book.T
            index = distances.argmin(-1)
            quant = F.embedding(index, book).clone()
            if self.training:
                with torch.no_grad():
                    counts = torch.bincount(index, minlength=self.size).float()
                    sums = torch.zeros_like(book).index_add_(0, index, residual)
                    self.counts[level].lerp_(counts, 1-self.decay)
                    self.sums[level].lerp_(sums, 1-self.decay)
                    unused = self.counts[level] < 1
                    choices = residual[torch.randint(len(z), (self.size,), device=z.device)]
                    self.sums[level][unused] = choices[unused]
                    self.counts[level][unused] = 1
                    n = self.counts[level].sum()
                    normal = n*(self.counts[level]+1e-5)/(n+self.size*1e-5)
                    self.books[level].copy_(self.sums[level]/normal[:, None])
            residual = residual - quant
            aggregate = aggregate + quant
            losses.append(F.mse_loss(z, aggregate.detach()))
            codes.append(index)
        if self.training:
            self.initialized.fill_(True)
        return z+(aggregate-z).detach(), torch.stack(losses).mean(), torch.stack(codes, -1)


class AdaptiveRQVAE(nn.Module):
    def __init__(self, input_dim, hidden=(1024, 512), latent=256, levels=3, size=2048, rank=64):
        super().__init__()
        self.encoder = MLP(input_dim, hidden, latent)
        self.decoder = MLP(latent, tuple(reversed(hidden)), input_dim)
        self.quantizer = ResidualEMA(levels, size, latent)
        self.adapters = nn.ModuleList()
        self.routers = nn.ModuleList([VariationalRouter(input_dim), VariationalRouter(input_dim)])
        self.rank = rank
        self.phase = 'universal'

    def set_phase(self, phase):
        if phase not in ('universal', 'domain'):
            raise ValueError('Invalid tokenizer phase')
        if phase == 'domain' and not self.adapters:
            self.adapters.extend([adapted_encoder(self.encoder, self.rank) for _ in range(2)])
        self.phase = phase
        self.encoder.requires_grad_(phase == 'universal')
        self.decoder.requires_grad_(phase == 'universal')
        self.routers.requires_grad_(phase == 'domain')
        self.quantizer.train(self.training and phase == 'universal')

    def latent(self, x, domains, phase):
        universal = self.encoder(x)
        if phase == 'universal':
            return universal, x.new_zeros(())
        fused, regularizer = universal.clone(), x.new_zeros(())
        for d in range(2):
            selected = domains == d
            if selected.any():
                gate, kl = self.routers[d](x[selected])
                fused[selected] = (1-gate)*universal[selected] + gate*self.adapters[d](x[selected])
                regularizer = regularizer + kl*selected.float().mean()
        return fused, regularizer

    def loss(self, x, domains, phase, beta=.25, vib=1e-3):
        self.set_phase(phase)
        z, kl = self.latent(x, domains, phase)
        q, commit, _ = self.quantizer(z)
        return F.mse_loss(self.decoder(q), x) + beta*commit + vib*kl

    @torch.no_grad()
    def codes(self, x, domains):
        self.eval()
        self.quantizer.eval()
        return self.quantizer(self.latent(x, domains, 'domain')[0])[2]
