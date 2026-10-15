"""LLM-only evaluation cache; Text v4 keeps its original implementation."""
from contextlib import contextmanager

import torch

from cdr_framework.text_model import TextSIRCDR


class LLMSIRCDR(TextSIRCDR):
    def __init__(self, catalog, graphs, config):
        super().__init__(catalog, graphs, config)
        self._evaluation_items = None

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
