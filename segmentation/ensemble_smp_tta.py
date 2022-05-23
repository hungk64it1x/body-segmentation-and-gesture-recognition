from argparse import ArgumentParser
import pandas as pd
from numpy import False_
from utilizes.config import load_cfg
from utilizes.dataloader import get_loader, get_test_loader
from utilizes.augment import Augmenter, TestAugmenter
import tqdm   
import torch
import torch.nn as nn
import cv2
from PIL import Image
from loguru import logger
from segmentation_models_pytorch.models import SegmentationModel
import os
from utilizes.utils import rle_encode
from glob import glob
from utilizes.visualize import save_img
from auxiliary.metrics.metrics import *
import numpy as np
import torch.nn.functional as F
import imageio
from datetime import datetime
from utilizes.refinement import refinement


def main():

    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=False, default="configs/default.yaml"
    )
    args = parser.parse_args()

    logger.info("Loading config")
    config_path = args.config
    config = load_cfg(config_path)

    gts = []
    prs = []
    
    dataset = config["dataset"]["test_data_path"][0].split("/")[-1]
    
    test_img_paths = []

    test_data_path = config["dataset"]["test_data_path"]
    for i in test_data_path:
        test_img_paths.extend(glob(os.path.join(i, "*")))

    test_img_paths.sort()


    test_augprams = config["test"]["augment"]
    test_transform = TestAugmenter(**test_augprams, img_size=config['train']['dataloader']['img_size'])
    
    test_loader = get_test_loader(
        test_img_paths,
        transforms=test_transform,
        **config["test"]["dataloader"],
        mode="test",
    )

    test_size = len(test_loader)
    
    logger.info('Evaluating with test size {}'.format(test_size))
    dev = config["test"]["dev"]
    logger.info("Loading model")
    model_prams = config["model"]
    backbone = config['model']['backbone']
    head = config['model']['head']
    model_name = f'{backbone}-{head}'
    
    if "save_dir" not in model_prams:
        save_dir = os.path.join("/mnt/data/hungpv/bd/checkpoint", model_name)
    else:
        save_dir = config["model"]["save_dir"]
    
    num_classes = config['model']['num_classes']
    models = []
    results = []
   
    for i, ckpt_path in enumerate(config['test']['checkpoint_ens']):
        # print(ckpt_path)
        
        model = SegmentationModel(
            encoder_name=backbone,
            decoder_name=head,
            classes=num_classes,
            encoder_weight=None
        )
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model_state_dict'])
        if dev == "cpu":
            model.cpu()
        else:
            model.cuda()
        models.append(model)
    # except:
    #     logger.info('Can not load model :( try find out sth ...')
    logger.info(f"Start testing {len(test_loader)} images in {dataset} dataset")
    df = pd.DataFrame(columns=['task', 'id', 'label'])
    csv_save = config['test']['csv_save']
    batch_size = config['test']['dataloader']['batchsize']
    # device = torch.device(dev)
    
    for i, pack in tqdm.tqdm(enumerate(test_loader, start=1)):
        results = []
        for model in models:
            model.eval()
            image, filename, img = pack
            size = (256, 256)
            name = os.path.splitext(filename[0])[0]
            
            ext = os.path.splitext(filename[0])[1]
            if dev == "cpu":
                image = image.cpu()
            else:
                image = image.cuda()

            for j in range(3):
                if j == 0:
                    out = model(image)
                    out = F.interpolate(out, size, mode='bilinear', align_corners=True)
                    out = nn.functional.interpolate(out, size, mode='bilinear', align_corners=True)
                    out = out.data.sigmoid().cpu().numpy().squeeze()
                    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
                if j == 1:
                    out_v = model(torch.flip(image, dims=(2, )))
                    out_v = F.interpolate(out_v, size, mode='bilinear', align_corners=True)
                    out_v = nn.functional.interpolate(out_v, size, mode='bilinear', align_corners=True)
                    out_v = out_v.data.sigmoid().cpu().numpy().squeeze()
                    out_v = (out_v - out_v.min()) / (out_v.max() - out_v.min() + 1e-8)
                    out_v = cv2.flip(out_v, 0)
                if j == 2:
                    out_h = model(torch.flip(image, dims=(3, )))
                    out_h = F.interpolate(out_h, size, mode='bilinear', align_corners=True)
                    out_h = nn.functional.interpolate(out_h, size, mode='bilinear', align_corners=True)
                    out_h = out_h.data.sigmoid().cpu().numpy().squeeze()
                    out_h = (out_h - out_h.min()) / (out_h.max() - out_h.min() + 1e-8)
                    out_h = cv2.flip(out_h, 1)
            res = (out + out_v + out_h) / 3
            results.append(res)

        res = 1*results[0] + 1.0*results[4] + 1.0*results[1] + 1.0*results[2] + 1.0*results[3]
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        
        res[res >= 0.005] = 1
        res[res < 0.005] = 0
        res = res.astype(np.uint8)
        res = refinement(res)
        df.loc[i, 'task'] = 0
        df.loc[i, 'id'] = filename[0]
        df.loc[i, 'label'] = rle_encode(res)
                            
    df.to_csv(f'./csv/sub1.csv', index=False)
    return df
        

        


if __name__ == "__main__":
    main()
