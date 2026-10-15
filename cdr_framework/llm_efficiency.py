"""Detached device-side statistics shared by LLM training and its disposable benchmark."""
import torch


class DeviceLossMeter:
    """Accumulate weighted losses without a CPU synchronization for each loss."""
    def __init__(self):
        self.names = None
        self.totals = None
        self.examples = 0

    def add(self, losses, count):
        if count < 1:
            raise ValueError('Loss count must be positive')
        if self.names is None:
            self.names = tuple(losses)
        if tuple(losses) != self.names:
            raise ValueError('Loss keys changed during accumulation')
        values = torch.stack([losses[name].detach() for name in self.names]).to(torch.float64)
        if self.totals is None:
            self.totals = values * count
        else:
            self.totals.add_(values, alpha=count)
        self.examples += count

    def means(self):
        if self.examples == 0:
            raise ValueError('No losses were accumulated')
        # One device-to-host transfer at the epoch boundary. Float64 mirrors the
        # previous Python-double weighted sums without retaining autograd graphs.
        values = (self.totals / self.examples).cpu().tolist()
        return dict(zip(self.names, values))
