import segmentation_models_pytorch as smp 
import torch
import torch.nn as nn

class SegmentationModel(nn.Module):
    def __init__(self, encoder_name='', decoder_name='', in_channels=3, classes=1, encoder_weight=None):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.in_channels = in_channels
        self.classes = classes
        self.encoder_weight = encoder_weight
        
        if str.lower(self.decoder_name) == 'unet':
            
            self.model = smp.Unet(encoder_name=self.encoder_name,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  encoder_weights=self.encoder_weight)
        elif str.lower(self.decoder_name) == 'unetplusplus':
            self.model = smp.UnetPlusPlus(encoder_name=self.encoder_name,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  encoder_weights=self.encoder_weight)
        elif str.lower(self.decoder_name) == 'deeplabv3':
            self.model = smp.DeepLabV3(encoder_name=self.encoder_name,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  encoder_weights=self.encoder_weight)
        elif str.lower(self.decoder_name) == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(encoder_name=self.encoder_name,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  encoder_weights=self.encoder_weight)
        elif str.lower(self.decoder_name) == 'deeplabv4':
            self.model = smp.DeepLabV4(encoder_name=self.encoder_name,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  encoder_weights=self.encoder_weight)
        elif str.lower(self.decoder_name) == 'pspnet':
            self.model = smp.PSPNet(encoder_name=self.encoder_name,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  encoder_weights=self.encoder_weight)
        elif str.lower(self.decoder_name) == 'manet':
            self.model = smp.MAnet(encoder_name=self.encoder_name,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  encoder_weights=self.encoder_weight)
        elif str.lower(self.decoder_name) == 'linknet':
            self.model = smp.Linknet(encoder_name=self.encoder_name,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  encoder_weights=self.encoder_weight)
        elif str.lower(self.decoder_name) == 'pan':
            self.model = smp.PAN(encoder_name=self.encoder_name,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  encoder_weights=self.encoder_weight)
        elif str.lower(self.decoder_name) == 'fpn':
            self.model = smp.FPN(encoder_name=self.encoder_name,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  encoder_weights=self.encoder_weight)
    def forward(self, x):
        out = self.model(x)
        return out