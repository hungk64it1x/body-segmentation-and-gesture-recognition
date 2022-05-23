import os
import math
import time

import random
import shutil
from pathlib import Path

from loguru import logger
import scipy as sp
import numpy as np
import pandas as pd
import albumentations as A
from sklearn import preprocessing

from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from functools import partial
import cv2
from PIL import Image
from trainer import *
import warnings 
warnings.filterwarnings('ignore')
from utils import *
import pandas as pd
df = pd.DataFrame(columns=['folder_id', 'folder_path', 'label'])

folder_list = []
folder_path = []
labels = []

for i in tqdm(range(CFG.NUM_CLASSES)):
    for id in os.listdir(f'{CFG.TRAIN_DIR}/{i}'):
        folder_list.append(id)
        folder_path.append(os.path.join(f'{CFG.TRAIN_DIR}/{i}', id))
        labels.append(i)
df['folder_id'] = folder_list
df['folder_path'] = folder_path
df['label'] = labels

folds = df.copy()
Fold = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.TARGET_COL])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
print(folds.groupby(['fold', CFG.TARGET_COL]).size())

def main():
    
    oof_df = pd.DataFrame(columns=['fold', 'score'])
    scores = []
    n_folds = []
    for fold in range(CFG.N_FOLDS):
        n_folds.append(fold)
        score = train_loop(folds, fold)
        scores.append(score)
        logger.info(f"========== fold: {fold} result ==========")
        logger.info(f'Best accuracy: {score}')
    oof = np.mean(scores)
    logger.info(f"========== CV ==========")
    logger.info("OOF: {}".format(oof))
    oof_df['fold'] = n_folds
    oof_df['score'] = scores
    oof_df.loc[CFG.N_FOLDS, ['fold']] = 'OOF'
    oof_df.loc[CFG.N_FOLDS, ['score']] = oof
    name_oof = 'oof_df.csv'
    oof_df.to_csv(os.path.join(CFG.OUTPUT_DIR, name_oof), index=False)
    
if __name__ == '__main__':
    main()

