from argparse import ArgumentParser
from utilizes.dataloader import get_loader
from utilizes.augment import Augmenter
from loguru import logger
from glob import glob
import torch
import torch.nn.functional as F
from utilizes.config import *
import os
from models.base import BaseModel
from models.heads import *
from models import CustomModel
from auxiliary import optimizers as optims
from auxiliary import schedulers as schedulers
from auxiliary import losses as losses
from tools.trainer import *
from datetime import datetime

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=False, default="configs/default.yaml"
    )
    args = parser.parse_args()

    logger.info("Loading config")
    config_path = args.config
    config = load_cfg(config_path)
    try:
        name_config = str(config_path).split('/')[-1].split('yaml')[0]
        print(name_config)
    except:
        logger.error('Config file path must be a yaml file, pls try again')
    try:
        dataset_name = config["dataset"].split('/')[-1]
    except:
        dataset_name = 'PolypDataset'
    time_now = str(datetime.now())
    
    
    os.makedirs(f'logs/{name_config}', exist_ok=True)
    logger.add(
        f'logs/{name_config}/train_{config["model"]["backbone"]}-{config["model"]["head"]}_{time_now}_{dataset_name}.log'
    )
    logger.info(f"Load config from {config_path}")
    logger.info(f"{config}")
    logger.info("Getting datapath")
    
    train_img_paths = []
    train_label_paths = []
    train_data_path = config["dataset"]["train_data_path"]
    if type(train_data_path) != list:
        train_img_paths = os.path.join(train_data_path, "training.tif")
        train_label_paths = os.path.join(train_data_path, "training_groundtruth.tif")
    else:
        for i in train_data_path:
            train_img_paths.extend(glob(os.path.join(i, "images", "*")))
            train_label_paths.extend(glob(os.path.join(i, "masks", "*")))
        train_img_paths.sort()
        train_label_paths.sort()
        logger.info(f"There are {len(train_img_paths)} images to train")
        
    is_val = config['train']['is_val']
    
    val_img_paths = []
    val_label_paths = []
    val_data_path = config["dataset"]["val_data_path"]
    if type(val_data_path) != list:
        val_img_paths = os.path.join(val_data_path, "testing.tif")
        val_label_paths = os.path.join(val_data_path, "testing_groundtruth.tif")
    else:
        for i in val_data_path:
            val_img_paths.extend(glob(os.path.join(i, "images", "*")))
            val_label_paths.extend(glob(os.path.join(i, "masks", "*")))
        val_img_paths.sort()
        val_label_paths.sort()
        if is_val:
            logger.info(f"There are {len(val_label_paths)} images to val")
        else:
            logger.info('Train model with no valid dataset')
    
    logger.info("Loading data")
    use_ddp = config['train']['ddp']
    if use_ddp:
        logger.info('Use distributed samplers')
    train_augprams = config["train"]["augment"]
    train_transform = Augmenter(**train_augprams, img_size=config['train']['dataloader']['img_size'])
    train_loader = get_loader(
        train_img_paths,
        train_label_paths,
        transforms=train_transform,
        **config["train"]["dataloader"],
        mode="train",
        use_ddp=use_ddp
    )
    logger.info(f"{len(train_loader)} batches to train")

    val_augprams = config["test"]["augment"]
    val_transform = Augmenter(**val_augprams, img_size=config['train']['dataloader']['img_size'])
    val_loader = get_loader(
        val_img_paths,
        val_label_paths,
        transforms=val_transform,
        **config["test"]["dataloader"],
        mode="val",
        use_ddp=use_ddp
    )
    
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
    pretrained = config['model']['pretrained']
    device = torch.device(config['dev'])
    if pretrained == '' or pretrained is None:
        logger.info('No pretrained, traning model from scratch..')
        model = CustomModel(backbone=str(backbone), decode=str(head), num_classes=num_classes, pretrained=pretrained)
    else:
        try:
            logger.info(f'Loading checkpoint from {pretrained} ...')
            model = CustomModel(backbone=str(backbone), decode=str(head), num_classes=num_classes, pretrained=pretrained)
            model.init_pretrained(pretrained)
        except:
            logger.info('Can not load pretrained model, train from scratch..')
            model = CustomModel(backbone=str(backbone), decode=str(head), num_classes=num_classes, pretrained=None)
        
    model = model.to(device)
    
    strat_fr = config['train']['start_from']
    if strat_fr != 0:
        restore_from = os.path.join(
            save_dir,
            f'{backbone}-{head}-{strat_fr}.pth',
            # e.g: ResNet50-UPerNet-2.pth
        )
        saved_state_dict = torch.load(restore_from)["model_state_dict"]
        lr = torch.load(restore_from)["lr"]
        model.load_state_dict(saved_state_dict, strict=False)

    params = model.parameters()
    
    opt_params = config["optimizer"]

    lr = opt_params["lr"]
    save_from = config['train']['save_from']

    # hardnet
    optimizer = optims.__dict__[opt_params["name"].lower()](params, lr / 8)
    scheduler = schedulers.__dict__[opt_params["scheduler"]](
        optimizer, config['train']["num_epochs"], config['train']["num_warmup_epoch"]
    )
    loss = losses.__dict__[opt_params["loss"]]()
    use_amp = config['train']['amp']
    try:
        use_multi_loss = config['train']['multi_loss']
    except :
        use_multi_loss = False
    if use_multi_loss:
        logger.info('Use deep supervision for training')
    if use_amp:
        logger.info('Training with mixed precision ...')
    else:
        logger.info('Training with FP32 ...')
    
    trainer = Trainer(
        model, model_name, optimizer, loss, scheduler, save_dir, save_from, logger, device, use_amp=use_amp, use_ddp=use_ddp, multi_loss=use_multi_loss
    )
    
    trainer.train_loop(
        train_loader,\
        val_loader,\
        config['train']['num_epochs'],\
        img_size=config['train']['dataloader']['img_size'],\
        size_rates=config['train']['size_rates'],\
        clip_grad=0.5,\
        is_val=is_val
    )
    
    
if __name__ == '__main__':
    main()
        
    

