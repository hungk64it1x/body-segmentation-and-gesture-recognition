from email.policy import strict
import torch
from torch import Tensor
from torch.nn import functional as F
from models.base import BaseModel
from models.heads import CondHead, FaPNHead, FCNHead, LawinHead, SegFormerHead, SFHead, UPerHead


class CustomModel(BaseModel):
    def __init__(self, backbone: str='ResT-S', decode: str='UPerHead', num_classes: int=1, pretrained=None) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = eval(str(decode))(self.backbone.channels, 128, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


if __name__ == '__main__':
    model = CustomModel('ResT-S', 19)
    # model.init_pretrained('checkpoints/backbones/rest/rest_small.pth')
    x = torch.zeros(2, 3, 352, 352)
    y = model(x)
    print(y.shape)
        

