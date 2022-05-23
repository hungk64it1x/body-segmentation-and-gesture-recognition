from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F

# from .focal_loss import FocalLoss
from .mcc_loss import MCC_Loss
# from pywick import losses as ls
from .tversky_loss import TverskyLoss
from .ssim import SSIM

class structure_loss(_Loss):
    def __init__(self):
        super(structure_loss, self).__init__()

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # print(pred.shape, mask.shape)  # (bs, 1, h, w) (bs, 1, h,w)
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=51, stride=1, padding=25) - mask
        )
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # wfocal = TverskyLoss(alpha=0.1, beta=0.9)(pred, mask)
        # wfocal = (wfocal * weit).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        # return (wfocal + wbce).mean()


        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


        # pred = torch.sigmoid(pred)
        # tp =(pred * mask * weit).sum(dim=(2, 3))
        # tn =((1 - pred) * (1 - mask) * weit).sum(dim=(2, 3))
        # fp =(pred * (1 - mask) * weit).sum(dim=(2, 3))
        # fn =((1 - pred) * mask * weit).sum(dim=(2, 3))
        # numerator = tp * tn - fp* fn
        # denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        # mcc = 1 - numerator.sum()/(denominator.sum() + 1.0)
        # return wbce.mean() + mcc


        # ssim_loss = SSIM(window_size=11,size_average=True)
        # ssim_out = 1 - ssim_loss(pred,mask)
        # return (wbce + wiou).mean() + ssim_out
