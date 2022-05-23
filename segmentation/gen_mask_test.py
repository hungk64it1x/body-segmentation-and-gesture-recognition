from argparse import ArgumentParser
import pandas as pd
from numpy import False_
from utilizes.config import load_cfg
from utilizes.dataloader import get_loader, get_test_loader
from utilizes.augment import Augmenter, TestAugmenter
import tqdm   
import torch
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
    try:
        model = SegmentationModel(
            encoder_name=backbone,
            decoder_name=head,
            classes=num_classes,
            encoder_weight=None
        )
        model.load_state_dict(torch.load(config['test']['checkpoint_dir'], map_location='cpu')['model_state_dict'])
    except:
        logger.info('Can not load model :( try find out sth ...')
        
    device = torch.device(dev)
    if dev == "cpu":
        model.cpu()
    else:
        model.cuda()
    model.eval()
    name_scenario = save_dir.split('/')[-1]
    checkpoint_names = os.listdir(save_dir)
    if config['test']['checkpoint_dir'] is not None:
        model_path = config['test']['checkpoint_dir']
    else:
        try:
            epoch_ckpts = [i.split('_')[-1].split('.')[0] for i in checkpoint_names]
            epoch_max = max(int(epoch_ckpts))
            model_path = os.path.join(save_dir, f'{model_name}_{epoch_max}.pth')

        except:
            model_path = os.path.join(save_dir, checkpoint_names[0])
    
    logger.info(f"Loading from {model_path}")
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device)["model_state_dict"]
        )
    except RuntimeError:
        model.load_state_dict(torch.load(model_path, map_location=device))


    if 'visualize_dir' in config['test']:
        visualize_dir = '/mnt/data/hungpv/polyps/visualize/default'
    else:
        visualize_dir = os.path.join(config['test']['visualize_dir'], name_scenario)
        os.makedirs(visualize_dir)
    mask_save = config['test']['mask_save']
    os.makedirs(mask_save, exist_ok=True)
    logger.info(f"Start generating mask ...")
    for i, pack in tqdm.tqdm(enumerate(test_loader, start=1)):
        
        image, filename, img = pack
        size = (256, 256)
        name = os.path.splitext(filename[0])[0]
        
        ext = os.path.splitext(filename[0])[1]
        if dev == "cpu":
            image = image.cpu()
        else:
            image = image.cuda()

        result = model(image)

        rles = []
        res = F.upsample(
            result, size=size, mode="bilinear", align_corners=False
        )

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res[res >= 0.005] = 255
        res[res < 0.005] = 0
        res = res.astype(np.uint8)
        res = refinement(res)
        Image.fromarray(res).save(os.path.join(mask_save, f'{name}{ext}'))
        


if __name__ == "__main__":
    main()
