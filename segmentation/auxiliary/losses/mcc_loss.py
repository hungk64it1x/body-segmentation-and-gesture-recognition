import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.nn.modules.loss import _Loss


# https://github.com/kakumarabhishek/MCC-Loss/blob/main/loss.py
class MCC_Loss(_Loss):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.
    Args:
        pred (torch.Tensor): 1-hot encoded predictions
        mask (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, pred: torch.Tensor, mask: torch.Tensor):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        pred = torch.sigmoid(pred)
        tp =(pred * mask).sum(dim=(2, 3))
        tn =((1 - pred) * (1 - mask)).sum(dim=(2, 3))
        fp =(pred * (1 - mask)).sum(dim=(2, 3))
        fn =((1 - pred) * mask).sum(dim=(2, 3))

        # tp = torch.sum(torch.mul(pred, mask))
        # tn = torch.sum(torch.mul((1 - pred), (1 - mask)))
        # fp = torch.sum(torch.mul(pred, (1 - mask)))
        # fn = torch.sum(torch.mul((1 - pred), mask))

        numerator = tp * tn - fp* fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator.sum()/(denominator.sum() + 1.0)

        # numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        # denominator = torch.sqrt(
        #     torch.add(tp, 1, fp)
        #     * torch.add(tp, 1, fn)
        #     * torch.add(tn, 1, fp)
        #     * torch.add(tn, 1, fn)
        # )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        # mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc