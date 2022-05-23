import torch
from torch.nn.modules.loss import _Loss

def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:,1,:,:,:] = (x == 1)
        xx[:,2,:,:,:] = (x == 2)
        xx[:,3,:,:,:] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:,0,:,:,:] = (x == 1)
        xx[:,1,:,:,:] = (x == 2)
        xx[:,2,:,:,:] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

class GeneralizedDiceLoss(_Loss):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor,eps=1e-5,weight_type='square') -> torch.Tensor:

        """
            Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
        """
        # print(output.shape,target.shape) # (bs,4 ,h,w,d), (bs, h,w,d)

        if target.dim() == 4:
            target[target == 4] = 3 # label [4] -> [3]
            target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

        output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
        target = flatten(target)[1:,...] # [class, N*H*W*D]

        target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels
        if weight_type == 'square':
            class_weights = 1. / (target_sum * target_sum + eps)
        elif weight_type == 'identity':
            class_weights = 1. / (target_sum + eps)
        elif weight_type == 'sqrt':
            class_weights = 1. / (torch.sqrt(target_sum) + eps)
        else:
            raise ValueError('Check out the weight_type :',weight_type)

        # print(class_weights)
        intersect = (output * target).sum(-1)
        intersect_sum = (intersect * class_weights).sum()
        denominator = (output + target).sum(-1)
        denominator_sum = (denominator * class_weights).sum() + eps

        loss1 = 2*intersect[0] / (denominator[0] + eps)
        loss2 = 2*intersect[1] / (denominator[1] + eps)
        loss3 = 2*intersect[2] / (denominator[2] + eps)
        # logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

        return 1 - 2. * intersect_sum / denominator_sum

