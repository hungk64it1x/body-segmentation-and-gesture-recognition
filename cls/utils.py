from config import CFG
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose
    )
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from contextlib import contextmanager
import os, cv2, random
import numpy as np
import torch
from collections import defaultdict, Counter
import time


def get_train_transforms():
    additional_targets = {f'image{i}':'image' for i in range(CFG.SEQ_LEN-1)}
    return A.Compose(
        [
            A.HorizontalFlip(p=0.3),
#             A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.2),
            A.RandomBrightness(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.3, p=0.3),
            A.Cutout(num_holes=12, max_h_size=10, max_w_size=10, p=0.4),
            
            A.Resize(width=CFG.IMAGE_SIZE, height=CFG.IMAGE_SIZE, always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ],
        additional_targets=additional_targets
    )

def get_valid_transforms():
    additional_targets = {f'image{i}':'image' for i in range(CFG.SEQ_LEN-1)}
    return A.Compose(
        [
            A.Resize(width=CFG.IMAGE_SIZE, height=CFG.IMAGE_SIZE, always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ],
        additional_targets=additional_targets
    )
	
def get_valid_tta_random_transforms():
    additional_targets = {f'image{i}':'image' for i in range(CFG.SEQ_LEN-1)}
    return A.Compose(
        [
            A.HorizontalFlip(p=0.2),
            A.Resize(width=CFG.IMAGE_SIZE, height=CFG.IMAGE_SIZE, always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ],
        additional_targets=additional_targets
    )

def get_valid_hflip_transforms():
    additional_targets = {f'image{i}':'image' for i in range(CFG.SEQ_LEN-1)}
    return A.Compose(
        [
            A.HorizontalFlip(p=1),
            A.Resize(width=CFG.IMAGE_SIZE, height=CFG.IMAGE_SIZE, always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ],
        additional_targets=additional_targets
    )

def get_valid_crop_transforms():
    additional_targets = {f'image{i}':'image' for i in range(CFG.SEQ_LEN-1)}
    return A.Compose(
        [
            A.RandomResizedCrop(224, 224),
            A.Resize(width=CFG.IMAGE_SIZE, height=CFG.IMAGE_SIZE, always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ],
        additional_targets=additional_targets
    )

def get_valid_vflip_transforms():
    additional_targets = {f'image{i}':'image' for i in range(CFG.SEQ_LEN-1)}
    return A.Compose(
        [
            A.VerticalFlip(p=1),
            A.Resize(width=CFG.IMAGE_SIZE, height=CFG.IMAGE_SIZE, always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ],
        additional_targets=additional_targets
    )
    
def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def init_progress_dict(metrics):
    progress_dict = dict()
    for metric in metrics:
        progress_dict[f'train_{metric}'] = []
        progress_dict[f'valid_{metric}'] = []
    return progress_dict

def log_to_progress_dict(progress_dict, metric_dict):
    for k, v in metric_dict.items():
        progress_dict[k].append(v)
       
    return progress_dict



def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.SEED)

def dfs_freeze(module):
    for name, child in module.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
        
def dfs_unfreeze(module):
    for name, child in module.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)
        
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def get_scheduler(optimizer):
    if CFG.SCHEDULER == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.FACTOR, patience=CFG.PATIENCE, verbose=True, eps=CFG.EPS)
    elif CFG.SCHEDULER == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_MAX, eta_min=CFG.MIN_LR, last_epoch=-1)
    elif CFG.SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.MIN_LR, last_epoch=-1)
    return scheduler