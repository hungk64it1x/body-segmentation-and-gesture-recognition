import pandas as pd
import cv2
import shutil
import os
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import argparse


seed = 42
train_dir = '../dataset/train'
folds_dir = '../dataset/5folds'
images_dir = os.path.join(train_dir, "images")
masks_dir = os.path.join(train_dir, "masks")
os.makedirs(folds_dir, exist_ok=True)
entire_images = os.listdir(images_dir)
df = pd.DataFrame(columns=['id', 'image_id', 'class'])
image_ids = []
ids = []
classes = []
for image_id in tqdm(entire_images):
    id = image_id.split('.')[0]
    ids.append(id)
    cls = image_id.split('_Subject')[0]
    image_ids.append(image_id)
    classes.append(cls)

df['image_id'] = image_ids
df['id'] = ids
df['class'] = classes
df['fold'] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for index, (train_ids, val_ids) in enumerate(skf.split(df, df['class'])):
    df.loc[val_ids, 'fold'] = index

if __name__ == '__main__':
    # print(df.head(3))
    # print(df.groupby(['fold', 'class']).size())
    for fold in df['fold'].tolist():
        new_dir = os.path.join(folds_dir, f'fold{fold}')
        new_images_dir = os.path.join(new_dir, 'images')
        new_masks_dir = os.path.join(new_dir, 'masks')
        os.makedirs(new_images_dir, exist_ok=True)
        os.makedirs(new_masks_dir, exist_ok=True)
    for i, row in tqdm(df.iterrows()):
        old_image_path = os.path.join(images_dir, row['image_id'])
        old_mask_path = os.path.join(masks_dir, row['image_id'])
        new_image_path = f'{folds_dir}/fold{row["fold"]}/images/{row["image_id"]}'
        new_mask_path = f'{folds_dir}/fold{row["fold"]}/masks/{row["image_id"]}'
        shutil.copy(old_image_path, new_image_path)
        shutil.copy(old_mask_path, new_mask_path)

