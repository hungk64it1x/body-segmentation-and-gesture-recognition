from segmentation_models_pytorch.models import SegmentationModel
import torch

encoder_name = 'timm-efficientnet-b0'
decoder_name = 'Unet'

if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256)
    model = SegmentationModel(encoder_name=encoder_name, decoder_name=decoder_name)
    print(model(x).shape)