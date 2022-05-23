import torch
import torch.nn as nn
import torch.nn.functional as F


class dice_loss(nn.Module):
    def __init__(self, n_classes=2):
        super(dice_loss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)).sum(dim=(2, 3))
        union = ((pred + mask)).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return wiou.mean()

        # target = target.float()
        # smooth = 1e-5
        # intersect = torch.sum(score * target)
        # y_sum = torch.sum(target * target)
        # z_sum = torch.sum(score * score)
        # loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # loss = 1 - loss
        # return loss

    # def forward(self, inputs, target, weight=None, softmax=False):
    #     if softmax:
    #         inputs = torch.softmax(inputs, dim=1)
    #     target = self._one_hot_encoder(target)
    #     if weight is None:
    #         weight = [1] * self.n_classes
    #     assert (
    #         inputs.size() == target.size()
    #     ), "predict {} & target {} shape do not match".format(
    #         inputs.size(), target.size()
    #     )
    #     class_wise_dice = []
    #     loss = 0.0
    #     for i in range(0, self.n_classes):
    #         dice = self._dice_loss(inputs[:, i], target[:, i])
    #         class_wise_dice.append(1.0 - dice.item())
    #         loss += dice * weight[i]
    #     return loss / self.n_classes
