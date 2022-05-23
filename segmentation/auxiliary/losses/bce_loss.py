from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F


class bce_loss(_Loss):
    def __init__(self):
        super(bce_loss, self).__init__()

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        return wbce.sum().mean()
