from argparse import ArgumentParser

from numpy import False_
from utilizes.config import load_cfg
from utilizes.dataloader import get_loader
from utilizes.augment import Augmenter
import tqdm   
import torch
from loguru import logger
from models import CustomModel
# from models import *
import os
from glob import glob
from utilizes.visualize import save_img
from auxiliary.metrics.metrics import *
import numpy as np
import torch.nn.functional as F
import imageio
from datetime import datetime


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
    test_mask_paths = []
    test_data_path = config["dataset"]["test_data_path"]
    for i in test_data_path:
        test_img_paths.extend(glob(os.path.join(i, "images", "*")))
        test_mask_paths.extend(glob(os.path.join(i, "masks", "*")))

    test_img_paths.sort()
    test_mask_paths.sort()

    test_augprams = config["test"]["augment"]
    test_transform = Augmenter(**test_augprams, img_size=config['train']['dataloader']['img_size'])
    
    test_loader = get_loader(
        test_img_paths,
        test_mask_paths,
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
        save_dir = os.path.join("/mnt/data/hungpv/polyps/checkpoint/KCECE", model_name)
    else:
        save_dir = config["model"]["save_dir"]
    
    num_classes = config['model']['num_classes']
    try:
        model = CustomModel(backbone=str(backbone), decode=str(head), num_classes=num_classes, pretrained=False)
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

    tp_all = 0
    fp_all = 0
    fn_all = 0
    tn_all = 0

    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    mean_F2 = 0
    mean_acc = 0
    mean_spe = 0
    mean_se = 0

    mean_precision_np = 0
    mean_recall_np = 0
    mean_iou_np = 0
    mean_dice_np = 0


    if 'visualize_dir' in config['test']:
        visualize_dir = '/mnt/data/hungpv/polyps/visualize/default'
    else:
        visualize_dir = os.path.join(config['test']['visualize_dir'], name_scenario)
        os.makedirs(visualize_dir)

    logger.info(f"Start testing {len(test_loader)} images in {dataset} dataset")

    for i, pack in tqdm.tqdm(enumerate(test_loader, start=1)):
        image, gt, filename, img = pack
        name = os.path.splitext(filename[0])[0]
        ext = os.path.splitext(filename[0])[1]
        gt = gt[0][0]
        gt = np.asarray(gt, np.float32).round()
        if dev == "cpu":
            image = image.cpu()
        else:
            image = image.cuda()

        result = model(image)

        res = F.upsample(
            result, size=gt.shape, mode="bilinear", align_corners=False
        )
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        overwrite = config["test"]["vis_overwrite"]
        vis_x = config["test"]["vis_x"]
        if config["test"]["visualize"]:
            save_img(
                os.path.join(
                    visualize_dir,
                    "PR_" + model_name,
                    "Hard",
                    name + ext,
                ),
                res.round() * 255,
                "cv2",
                overwrite,
            )
            save_img(
                os.path.join(
                    visualize_dir,
                    "PR_" + model_name,
                    "Soft",
                    name + ext,
                ),
                res * 255,
                "cv2",
                overwrite,
            )
            mask_img = (
                np.asarray(img[0])
                + vis_x
                * np.array(
                    (
                        np.zeros_like(res.round()),
                        res.round(),
                        np.zeros_like(res.round()),
                    )
                ).transpose((1, 2, 0))
                + vis_x
                * np.array(
                    (gt, np.zeros_like(gt), np.zeros_like(gt))
                ).transpose((1, 2, 0))
            )
            mask_img = mask_img[:, :, ::-1]
            save_img(
                os.path.join(
                    visualize_dir,
                    "GT_PR_" + model_name,
                    name + ext,
                ),
                mask_img,
                "cv2",
                overwrite,
            )

        pr = res.round()
        prs.append(pr)
        gts.append(gt)

        tp = np.sum(gt * pr)
        fp = np.sum(pr) - tp
        fn = np.sum(gt) - tp
        tn = np.sum((1 - pr) * (1 - gt))

        tp_all += tp
        fp_all += fp
        fn_all += fn
        tn_all += tn

        mean_precision += precision_m(gt, pr)
        mean_recall += recall_m(gt, pr)
        mean_iou += jaccard_m(gt, pr)
        mean_dice += dice_m(gt, pr)
        mean_F2 += (5 * precision_m(gt, pr) * recall_m(gt, pr)) / (
            4 * precision_m(gt, pr) + recall_m(gt, pr)
        )
        mean_acc += (tp + tn) / (tp + tn + fp + fn)
        mean_se += tp / (tp + fn)
        mean_spe += tn / (tn + fp)


    mean_precision /= len(test_loader)
    mean_recall /= len(test_loader)
    mean_iou /= len(test_loader)
    mean_dice /= len(test_loader)
    mean_F2 /= len(test_loader)
    mean_acc /= len(test_loader)
    mean_se /= len(test_loader)
    mean_spe /= len(test_loader)

    logger.info(
        "Macro scores: Dice: {:.3f} | IOU: {:.3f} | Precision: {:.3f} | Recall: {:.3f} | F2: {:.3f} | Se: {:.3f} | Spe {:.3f} | Acc: {:.3f}".format(
            mean_dice,
            mean_iou,
            mean_precision,
            mean_recall,
            mean_F2,
            mean_se,
            mean_spe,
            mean_acc
        )
    )

    precision_all = tp_all / (tp_all + fp_all + 1e-07)
    recall_all = tp_all / (tp_all + fn_all + 1e-07)
    dice_all = 2 * precision_all * recall_all / (precision_all + recall_all)
    iou_all = (
        recall_all
        * precision_all
        / (recall_all + precision_all - recall_all * precision_all)
    )
    logger.info(
        "Micro scores: Dice: {:.3f} | IOU: {:.3f} | Precision: {:.3f} | Recall: {:.3f}".format(
            dice_all,
            iou_all, 
            precision_all, 
            recall_all
            
        )
    )

    # from utils.metrics import get_scores_v1, get_scores_v2

    # get_scores_v1(gts, prs, logger)


    return gts, prs


if __name__ == "__main__":
    main()
