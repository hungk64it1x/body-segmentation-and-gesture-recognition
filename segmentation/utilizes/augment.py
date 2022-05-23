import albumentations as A
from albumentations.augmentations.transforms import Normalize
import matplotlib.pyplot as plt
from torchvision import transforms
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from albumentations.core.composition import Compose, OneOf
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    Resize,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout,
    ColorJitter,
)
import torch.nn as nn


class Augmenter(nn.Module):
    def __init__(
        self,
        prob=0,
        Flip_prob=0,
        HueSaturationValue_prob=0,
        RandomBrightnessContrast_prob=0,
        crop_prob=0,
        randomrotate90_prob=0,
        elastictransform_prob=0,
        gridistortion_prob=0,
        opticaldistortion_prob=0,
        verticalflip_prob=0,
        horizontalflip_prob=0,
        randomgamma_prob=0,
        CoarseDropout_prob=0,
        RGBShift_prob=0,
        MotionBlur_prob=0,
        MedianBlur_prob=0,
        GaussianBlur_prob=0,
        GaussNoise_prob=0,
        ChannelShuffle_prob=0,
        ColorJitter_prob=0,
        img_size=352
    ):
        super().__init__()

        self.prob = prob
        self.img_size = img_size
        self.randomrotate90_prob = randomrotate90_prob
        self.elastictransform_prob = elastictransform_prob

        self.transforms = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensorV2()
        ])
        self.transforms = A.Compose(
            [
                A.RandomRotate90(p=randomrotate90_prob),
                A.Flip(p=Flip_prob),
                A.HueSaturationValue(p=HueSaturationValue_prob),
                A.RandomBrightnessContrast(p=RandomBrightnessContrast_prob),
                
                OneOf(
                    [
                        A.RandomResizedCrop(self.img_size, self.img_size, p=0.3),
                        A.CenterCrop(220, 220, p=0.5),
                    ],
                    p=crop_prob,
                ),

                ElasticTransform(
                    p=elastictransform_prob,
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                ),
                GridDistortion(p=gridistortion_prob),
                OpticalDistortion(
                    p=opticaldistortion_prob, distort_limit=2, shift_limit=0.5
                ),
                VerticalFlip(p=verticalflip_prob),
                HorizontalFlip(p=horizontalflip_prob),
                RandomGamma(p=randomgamma_prob),
                RGBShift(p=RGBShift_prob),
                MotionBlur(p=MotionBlur_prob, blur_limit=7),
                MedianBlur(p=MedianBlur_prob, blur_limit=9),
                GaussianBlur(p=GaussianBlur_prob, blur_limit=9),
                GaussNoise(p=GaussNoise_prob),
                ChannelShuffle(p=ChannelShuffle_prob),
                CoarseDropout(
                    p=CoarseDropout_prob, max_holes=8, max_height=32, max_width=32
                ),
                ColorJitter(p=ColorJitter_prob),

                
            ],
            p=self.prob,
            
        )

    def forward(self, image, mask, softlabel=None):
        if softlabel is None:
            result = self.transforms(image=image, mask=mask)
        else:
            result = self.transforms(image=image, mask=mask, softlabel=softlabel)

        return result
    
class TestAugmenter(nn.Module):
    def __init__(
        self,
        prob=0,
        Flip_prob=0,
        HueSaturationValue_prob=0,
        RandomBrightnessContrast_prob=0,
        crop_prob=0,
        randomrotate90_prob=0,
        elastictransform_prob=0,
        gridistortion_prob=0,
        opticaldistortion_prob=0,
        verticalflip_prob=0,
        horizontalflip_prob=0,
        randomgamma_prob=0,
        CoarseDropout_prob=0,
        RGBShift_prob=0,
        MotionBlur_prob=0,
        MedianBlur_prob=0,
        GaussianBlur_prob=0,
        GaussNoise_prob=0,
        ChannelShuffle_prob=0,
        ColorJitter_prob=0,
        img_size=352
    ):
        super().__init__()

        self.prob = prob
        self.img_size = img_size
        self.randomrotate90_prob = randomrotate90_prob
        self.elastictransform_prob = elastictransform_prob

        self.transforms = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensorV2()
        ])
        self.transforms = A.Compose(
            [
                A.RandomRotate90(p=randomrotate90_prob),
                A.Flip(p=Flip_prob),
                A.HueSaturationValue(p=HueSaturationValue_prob),
                A.RandomBrightnessContrast(p=RandomBrightnessContrast_prob),
                
                OneOf(
                    [
                        A.RandomResizedCrop(self.img_size, self.img_size, p=0.3),
                        A.CenterCrop(220, 220, p=0.5),
                    ],
                    p=crop_prob,
                ),
                # A.RandomResizedCrop(self.img_size, self.img_size, p=0.3),
                ElasticTransform(
                    p=elastictransform_prob,
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                ),
                GridDistortion(p=gridistortion_prob),
                OpticalDistortion(
                    p=opticaldistortion_prob, distort_limit=2, shift_limit=0.5
                ),
                VerticalFlip(p=verticalflip_prob),
                HorizontalFlip(p=horizontalflip_prob),
                RandomGamma(p=randomgamma_prob),
                RGBShift(p=RGBShift_prob),
                MotionBlur(p=MotionBlur_prob, blur_limit=7),
                MedianBlur(p=MedianBlur_prob, blur_limit=9),
                GaussianBlur(p=GaussianBlur_prob, blur_limit=9),
                GaussNoise(p=GaussNoise_prob),
                ChannelShuffle(p=ChannelShuffle_prob),
                CoarseDropout(
                    p=CoarseDropout_prob, max_holes=8, max_height=32, max_width=32
                ),
                ColorJitter(p=ColorJitter_prob),
                
            ],
            p=self.prob,
            
        )

    def forward(self, image,softlabel=None):
        if softlabel is None:
            result = self.transforms(image=image)
        else:
            result = self.transforms(image=image)

        return result
    
class NoAugmenter(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.transforms = A.Compose([
            Resize(self.img_size, self.img_size, p=1),
            Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    def forward(self, image, mask, softlabel=None):
        
        if(softlabel is None):
            result = self.transforms(image=image, mask=mask)
        else:
            result = self.transforms(image=image, mask=mask, softlabel=softlabel)

        return result
        
