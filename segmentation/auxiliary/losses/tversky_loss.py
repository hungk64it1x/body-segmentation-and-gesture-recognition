

from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F



class TverskyLoss(_Loss):
    """Computes the Tversky loss [1].
        https://github.com/achaiah/pywick/blob/master/pywick/losses.py
        Args:
            :param alpha: controls the penalty for false positives.
            :param beta: controls the penalty for false negatives.
            :param eps: added to the denominator for numerical stability.
        Returns:
            tversky_loss: the Tversky loss.
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
        References:
            [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha=0.2, beta=0.8, eps=1e-7, **kwargs):
        super(TverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
            :param targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            :return: loss
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[targets.squeeze(1).long()]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[targets.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, logits.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_loss = (num / (denom + self.eps)).mean()

        return 1 - tversky_loss