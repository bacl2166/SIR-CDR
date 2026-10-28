"""LLM-only evaluation cache; Text v4 keeps its original implementation."""
from contextlib import contextmanager

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from cdr_framework.text_model import TextSIRCDR


class LLMSIRCDR(TextSIRCDR):
    def __init__(self, catalog, graphs, config):
        super().__init__(catalog, graphs, config)
        self._evaluation_items = None

    def sequence(self, base, shared, specific, ids, lengths, gru):
        # Same operations as TextSIRCDR.sequence, with separate CPU packing
        # lengths and device mask lengths. Evaluation's GPU lengths remain valid.
        gate = torch.sigmoid(self.structural_gate(torch.cat([base[ids], shared[ids], specific[ids]], -1)))
        inputs = self.dropout(base[ids] + gate * shared[ids] + (1 - gate) * specific[ids])
        packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, final = gru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=ids.shape[1])
        mask = torch.arange(ids.shape[1], device=ids.device)[None] < lengths.to(ids.device)[:, None]
        scores = self.attention(encoded).squeeze(-1).masked_fill(~mask, -torch.inf)
        pooled = (scores.softmax(1)[..., None] * encoded).sum(1)
        return 0.5 * (pooled + final[-1])

    def item_states(self):
        if self._evaluation_items is not None:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError('Evaluation catalog cannot be used in training or autograd')
            return self._evaluation_items
        return super().item_states()

    @contextmanager
    def evaluation_catalog(self):
        """Reuse item states only within a fixed-weight, no-grad evaluation phase."""
        if self.training:
            raise RuntimeError('Call model.eval() before caching the evaluation catalog')
        if self._evaluation_items is not None:
            raise RuntimeError('Nested evaluation catalog scopes are not supported')
        with torch.no_grad():
            self._evaluation_items = super().item_states()
            try:
                yield
            finally:
                self._evaluation_items = None

    def train(self, mode=True):
        if mode and self._evaluation_items is not None:
            raise RuntimeError('Cannot enter training with an active evaluation catalog')
        return super().train(mode)

    def load_state_dict(self, *args, **kwargs):
        if self._evaluation_items is not None:
            raise RuntimeError('Cannot load weights with an active evaluation catalog')
        return super().load_state_dict(*args, **kwargs)
