"""Paper-guided decoder reconstruction: universal LoRA experts + domain LoRA.

Upstream publishes no recommendation decoder. Expert mixing is implemented at
each adapted q/v projection; user-level routing mixes token probabilities as in
paper Eq.11. These implementation choices are disclosed, not called author code.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .tokenizer import VariationalRouter


class ExpertLinear(nn.Module):
    def __init__(self, base, experts, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        a, b = base.in_features, base.out_features
        self.universal_down = nn.Parameter(torch.empty(experts, rank, a))
        self.universal_up = nn.Parameter(torch.zeros(experts, b, rank))
        self.expert_gate = nn.Linear(a, experts)
        self.domain_down = nn.Parameter(torch.empty(rank, a))
        self.domain_up = nn.Parameter(torch.zeros(b, rank))
        for p in (self.universal_down, self.domain_down):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.scale, self.domain_active = alpha/rank, False

    def forward(self, x):
        xf = x.float()
        hidden = torch.einsum('...i,eri->...er', xf, self.universal_down)
        experts = torch.einsum('...er,eor->...eo', hidden, self.universal_up)
        delta = (experts*self.expert_gate(xf).softmax(-1)[..., None]).sum(-2)
        if self.domain_active:
            delta = delta + F.linear(F.linear(xf, self.domain_down), self.domain_up)
        return self.base(x) + (delta*self.scale).to(x.dtype)


class GenCDRDecoder(nn.Module):
    def __init__(self, backbone, levels=3, size=2048, experts=4, rank=64, alpha=128):
        super().__init__()
        self.backbone, self.levels, self.size = backbone, levels, size
        backbone.requires_grad_(False)
        self._frozen_parameter_ids = {id(p) for p in backbone.parameters()}
        projections = [(name, child) for name, child in backbone.named_modules()
                       if name.split('.')[-1] in ('q_proj', 'v_proj') and isinstance(child, nn.Linear)]
        if not projections:
            raise ValueError('Backbone has no supported attention projections')
        for name, child in projections:
            parent = backbone.get_submodule(name.rsplit('.', 1)[0])
            setattr(parent, name.split('.')[-1], ExpertLinear(child, experts, rank, alpha).to(child.weight.device))
        width = backbone.config.hidden_size
        self.embedding = nn.Embedding(4+levels*size, width, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        self.user_router = VariationalRouter(width)
        self.phase = None
        self.set_phase('universal')

    def set_phase(self, phase):
        if phase not in ('universal', 'domain'):
            raise ValueError('Invalid recommendation phase')
        self.phase = phase
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.embedding.requires_grad_(phase == 'universal')
        self.user_router.requires_grad_(phase == 'domain')
        for module in self.backbone.modules():
            if isinstance(module, ExpertLinear):
                module.universal_down.requires_grad_(phase == 'universal')
                module.universal_up.requires_grad_(phase == 'universal')
                module.expert_gate.requires_grad_(phase == 'universal')
                module.domain_down.requires_grad_(phase == 'domain')
                module.domain_up.requires_grad_(phase == 'domain')

    def portable_state(self):
        # Never copy the frozen pretrained model or its vocabulary to every job.
        return {n: p.detach().cpu() for n, p in self.named_parameters()
                if id(p) not in self._frozen_parameter_ids}

    def load_portable(self, payload):
        expected = set(self.portable_state())
        if set(payload) != expected:
            raise RuntimeError('Decoder checkpoint parameter identities differ')
        self.load_state_dict(payload, strict=False)

    def _hidden(self, ids, mask, domain):
        for module in self.backbone.modules():
            if isinstance(module, ExpertLinear):
                module.domain_active = domain
        dtype = next(self.backbone.parameters()).dtype
        position_ids = (mask.cumsum(-1)-1).clamp_min(0)
        out = self.backbone(inputs_embeds=self.embedding(ids).to(dtype), attention_mask=mask,
            position_ids=position_ids, use_cache=False, return_dict=True)
        return out.last_hidden_state.float()

    def _probabilities(self, ids, mask, gather, depths, context, context_mask, phase):
        def run(domain):
            hidden = self._hidden(ids, mask, domain)
            state = hidden.gather(1, gather[..., None].expand(-1, -1, hidden.shape[-1]))
            # A single semantic vocabulary, disjoint per quantization depth.
            all_logits = F.linear(state, self.embedding.weight[4:])
            logits = all_logits.reshape(*state.shape[:2], self.levels, self.size)
            logits = logits.gather(2, depths[..., None, None].expand(-1, -1, 1, self.size)).squeeze(2)
            return logits.log_softmax(-1)
        universal = run(False)
        if phase == 'universal':
            return universal, universal.new_zeros(())
        specific = run(True)
        h = (self.embedding(context)*context_mask[..., None]).sum(1)/context_mask.sum(1, keepdim=True)
        gate, kl = self.user_router(h)
        gate = gate.clamp(1e-6, 1-1e-6)[:, None]
        mixed = torch.logaddexp(universal+torch.log1p(-gate), specific+gate.log())
        return mixed, kl

    def loss(self, tokens, mask, labels, phase, vib=1e-3):
        self.set_phase(phase)
        lengths = mask.sum(1)
        rows = []
        offsets = torch.arange(self.levels, device=tokens.device)*self.size+4
        for row, length, label in zip(tokens, lengths.tolist(), labels):
            rows.append(torch.cat([row[:length], (label+offsets)[:-1]]))
        ids = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        full_lengths = lengths+self.levels-1
        attn = torch.arange(ids.shape[1], device=ids.device)[None] < full_lengths[:, None]
        depths = torch.arange(self.levels, device=ids.device)[None].expand(len(ids), -1)
        gather = lengths[:, None]-1+depths
        logs, kl = self._probabilities(ids, attn.long(), gather, depths, tokens, mask, phase)
        return -logs.gather(-1, labels[..., None]).mean() + vib*kl

    def next_log_probs(self, tokens, mask, prefixes, phase):
        depths_list = [len(p) for p in prefixes]
        if len(set(depths_list)) != 1 or depths_list[0] >= self.levels:
            raise ValueError('Generation batch must have the same valid prefix depth')
        rows = []
        for row, length, prefix in zip(tokens, mask.sum(1).tolist(), prefixes):
            encoded = torch.tensor([4+k*self.size+c for k, c in enumerate(prefix)], device=row.device)
            rows.append(torch.cat([row[:length], encoded.long()]))
        ids = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        lengths = torch.tensor([len(row) for row in rows], device=ids.device)
        attn = (torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]).long()
        depths = torch.full((len(ids), 1), depths_list[0], device=ids.device, dtype=torch.long)
        logs, _ = self._probabilities(ids, attn, lengths[:, None]-1, depths, tokens, mask, phase)
        return logs[:, 0]
