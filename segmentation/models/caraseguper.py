import torch
from torch import Tensor
from torch.nn import functional as F
from models.base import BaseModel
from models.heads import UPerHead
from models.lib.conv_layer import Conv, BNPReLU
from models.lib.axial_atten import AA_kernel
from models.lib.context_module import CFPModule

class CaraSegUPer(BaseModel):
    def __init__(self, backbone: str = 'PVTv2-B3', num_classes: int = 1, pretrained=None) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = UPerHead(self.backbone.channels, 768, num_classes)
        self.apply(self._init_weights)

        self.CFP_1 = CFPModule(128, d = 8)
        self.CFP_2 = CFPModule(320, d = 8)
        self.CFP_3 = CFPModule(512, d = 8)

        self.ra1_conv1 = Conv(128,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(320,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(512,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        x1 = y[0]  #  64x88x88
        x2 = y[1]  # 128x44x44
        x3 = y[2]  # 320x22x22
        x4 = y[3]  # 512x11x11

        decoder_1 = self.decode_head(y)   # 4x reduction in image size
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4) 
        cfp_out_1 += x4
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(cfp_out_1)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(x3)
        cfp_out_2 += x3
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(cfp_out_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2)
        cfp_out_3 += x2
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(cfp_out_3)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1
    
if __name__ == '__main__':
    model = CaraSegUPer()
    x = torch.rand(1, 3, 352, 352)
    out = model(x)
    print(out.shape)