# Copied from
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/metrics/perplexity.py
import torch
from torch import Tensor
from torchmetrics import Metric

# pylint: disable=no-member
class Perplexity(Metric):
    """Perplexity measures how well a Language Model represents a text.
    Calculated as exp(avg(nll)) where nll = negative log-likelihood i.e.
    the output of CrossEntropyLoss.
    """
    is_differentiable = True
    full_state_update = False
    higher_is_better = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(self):
        super().__init__()
        self.add_state("total_log_probs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction=None)

    def update(self, logits: Tensor, labels: Tensor, loss: torch.float32) -> None:
        """Update state with predictions and targets.
        Args:
            logits: Predictions from model in shape (batch_size, seq_len, vocab_size)
            labels: Ground truth values in shape (batch_size, seq_len)
        """
        count = labels.numel()
        if loss is None:
            # note: set `reduction=None` so returned `loss` is sum of the batch
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        else:
            # the passed loss is from model output which has already been reduced
            loss = loss * count
        self.total_log_probs += loss
        self.count += count

    def compute(self) -> Tensor:
        """Compute perplexity over state.
        Returns:
            Perplexity
        """
        return torch.exp(self.total_log_probs / self.count)
