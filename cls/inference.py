import os
import math
import time
from datetime import datetime
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
from loguru import logger
import scipy as sp
import numpy as np
import pandas as pd
import albumentations as A
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from functools import partial
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import timm
import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
from tqdm import tqdm
import os
from utils import get_train_transforms, get_valid_transforms, get_valid_tta_random_transforms
from trainer import test_fn, infer_fn
from model import GestureSeqModel
from dataset import GestureTestDataset

train_transform = get_train_transforms()
valid_transform = get_valid_transforms()
valid_tta_random = get_valid_tta_random_transforms()

class CFG:
    TRAIN_DIR = './dataset/frame_dataset/train'
    TEST_DIR = './dataset/private_test/frame_dataset'
    MASK_TRAIN_DIR = './dataset/segment_dataset/train'
    MASK_TEST_DIR = './dataset/private_test/segment_dataset'
    NUM_CLASSES = 10
    IMAGE_SIZE = 256
    N_FOLDS = 5
    SEED = 42
    TARGET_COL = 'label'
    SEQ_LEN = 16
    BATCH_SIZE = 8
    NUM_EPOCHS = 12
    NUM_WORKERS = 4
    FRAMES = np.arange(0, SEQ_LEN, 1).tolist()
    LABELS = ['VAR', 'look_at_me', 'reverse_signal', 'scratch', 'up_down', 'hand_fand', 'peekaboo', 'scissor', 'typing', 'wave_hand']
    BACKBONE = 'dm_nfnet_f0'
    DROP_RATE = 0.1
    DROP_PATH_RATE = 0.0
    PRETRAINED = True
    LSTM_HIDDEN_SIZE = 128
    LSTM_LAYERS = 1
    LSTM_DROP = 0.0
    SCHEDULER = 'CosineAnnealingWarmRestarts'
    FACTOR = 0.2 
    PATIENCE = 4 
    WEIGHT_DECAY = 1e-6
    EPS = 1e-6 
    T_MAX = 12 
    T_0 = 10 
    BASE_LR = 1e-4
    MIN_LR = 1e-8
    WARM_UP = 3
    TTA = True
    NUM_TTA = 5

    DEVICE = 'cuda'
    OUTPUT_DIR = f'./cls/{BACKBONE}'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    CHECKPOINT_DIR = './checkpoint'
    models = [
        'dm_nfnet_f0',
        # 'eca_nfnet_l0',
        'eca_nfnet_l1',
        # 'eca_nfnet_l2'
        # 'resnext101_32x8d'
    ]
    size = {
        
        "dm_nfnet_f0": 256,
        "eca_nfnet_l0": 256, 
        "eca_nfnet_l1": 256,
        "eca_nfnet_l2": 256,
        'resnext101_32x8d': 256,
    }
    trn_fold = {  
        "dm_nfnet_f0": {
#             "best": [0, 1, 2, 3, 4],
            "best": [0, 1, 2],
            "final": [],
            "weight": [0.5, 0.5, 0.8, 0.05, 0.05]
        },
        "eca_nfnet_l0": {
#             "best": [0, 1, 2, 3, 4],
            "best": [2, 3],
            "final": [],
            "weight": [0.7, 0.3, 0.7, 0.05, 0.05]
        },
        "eca_nfnet_l1": {"best": [2, 3], "final": [], "weight": [0.6, 0.4, 0.2, 0.2, 0.05]},
        "eca_nfnet_l2": {"best": [1, 4], "final": [], "weight": [0.5, 0.1, 0.7, 0.05, 0.05]},
        "resnext101_32x8d": {"best": [3], "final": [], "weight": [0.7, 0.3, 0.2, 0.2, 0.05]}
    }
    total_weight = [3, 2, 0.5, 0.5]
    # total_weight = [1]
    
if __name__ =='__main__':
    final_result = []
    test_folder_list = os.listdir(CFG.TEST_DIR)
    test_folder_list = [int(i) for i in test_folder_list]
    test_folder_list.sort()

    for ind, model_name in enumerate(CFG.models):
        logger.info("====== Model {} fold {} ======".format(model_name, CFG.trn_fold[model_name]['best']))
        model = GestureSeqModel(model_name, backbone_pretrained=False, lstm_dim=CFG.LSTM_HIDDEN_SIZE, lstm_layers=CFG.LSTM_LAYERS, n_classes=CFG.NUM_CLASSES)
        for i, fold in tqdm(enumerate(CFG.trn_fold[model_name]['best'])):
            temp_result = []
            fold_ckpt_path = os.path.join(CFG.CHECKPOINT_DIR, f'{model_name}/{model_name}_fold{fold}_best.pth')
            
            test_dataset = GestureTestDataset(CFG.TEST_DIR, CFG.MASK_TEST_DIR, test_folder_list, CFG.FRAMES, transform=valid_tta_random)
            test_loader = DataLoader(test_dataset, 
                                batch_size=CFG.BATCH_SIZE, 
                                shuffle=False, 
                                num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False)
            inf = infer_fn(test_loader, model, checkpoint_dir=fold_ckpt_path, device='cuda:0')
            temp_result.append(inf * CFG.trn_fold[model_name]['weight'][i])
        temp_model_result = sum(temp_result) / (i + 1)
        final_result.append(temp_model_result * CFG.total_weight[ind])
    result = np.argmax(sum(final_result) / len(CFG.total_weight), axis=1)
    submit = pd.DataFrame(columns=['task', 'id', 'label'])
    string_labels = [CFG.LABELS[i] for i in result]
    for i, cls in tqdm(enumerate(string_labels)):
        submit.loc[i, 'task'] = 1
        submit.loc[i, 'id'] = str(i) + '.avi'
        submit.loc[i, 'label'] = cls
    submit.to_csv('./csv/sub2.csv', index=False)
    


