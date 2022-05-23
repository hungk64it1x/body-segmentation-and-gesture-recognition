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
    models = []
    results = []
    mask_save = config['test']['mask_save']
    os.makedirs(mask_save, exist_ok=True)
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
            result = model(image)

            res = F.upsample(
                result, size=size, mode="bilinear", align_corners=False
            )
            
            results.append(res)
            
            
        # print(len(results))
        # res = sum(r for r in results) / len(results)
        res = 0.2*results[0] + 0.2*results[4] + 0.2*results[1] + 0.2*results[2] + 0.2*results[3]
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        
        res[res >= 0.005] = 255
        res[res < 0.005] = 0
        res = res.astype(np.uint8)
        res = refinement(res)
        Image.fromarray(res).save(os.path.join(mask_save, f'{name}{ext}'))
        



if __name__ == "__main__":
    main()
