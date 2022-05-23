from utils import *
from tqdm import tqdm
import torch
import numpy as np
from model import *
from dataset import GestureDataset
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from loguru import logger
from torch.optim import Adam, SGD
from warmup_scheduler import GradualWarmupScheduler

train_transform = get_train_transforms()
valid_transform = get_valid_transforms()

seed_torch(seed=CFG.SEED)

def train_valid_fn(dataloader,model, criterion, scaler, optimizer=None,device='cuda:0',scheduler=None,
                   epoch=0,mode='train', metric='acc'):
    '''Perform model training'''
    if(mode=='train'):
        model.train()
    elif(mode=='valid'):
        model.eval()
    else:
        raise ValueError('No such mode')
        
    loss_score = AverageMeter()
    
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    all_predictions = []
    all_labels = []
    for i, batch in tk0:
        if(mode=='train'):
            optimizer.zero_grad()
            
        # input, gt
        voxels, labels = batch
        voxels = voxels.to(device)
        labels = labels.to(device)

        # prediction
        with torch.cuda.amp.autocast():
            logits = model(voxels)
#             logits = logits.view(-1)
            probs = logits.softmax(1)
            preds = probs.argmax(1).detach().cpu().numpy()
            # compute loss
            loss = criterion(logits, labels)
        
        if(mode=='train'):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        loss_score.update(loss.detach().cpu().item(), dataloader.batch_size)

        # append for metric calculation
        all_predictions.append(preds)
        all_labels.append(labels.detach().cpu().numpy())
        
        if(mode=='train'):
            tk0.set_postfix(Loss_Train=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])
        elif(mode=='valid'):
            tk0.set_postfix(Loss_Valid=loss_score.avg, Epoch=epoch)
        
        del batch, voxels, labels, logits, probs, loss
        torch.cuda.empty_cache()

    if(mode=='train'):
        if(scheduler.__class__.__name__ == 'CosineAnnealingWarmRestarts'):
            scheduler.step(epoch=epoch)
        elif(scheduler.__class__.__name__ == 'ReduceLROnPlateau'):
            scheduler.step(loss_score.avg)

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    if(metric == 'acc'):
        acc = get_score(all_labels, all_predictions)
        return loss_score.avg, acc 
    
    return loss_score.avg


def train_loop(folds, fold):
    logger.add(
        f'{CFG.OUTPUT_DIR}/logs/train_{CFG.BACKBONE}_{str(datetime.now())}_{fold}.log',
        rotation="10 MB",
    )
    logger.info('====== Start training fold {} ======'.format(fold))
    train_df = folds[folds['fold'] != fold]
    valid_df = folds[folds['fold'] == fold]
    train_folder_ids = train_df['folder_id'].tolist()
    valid_folder_ids = valid_df['folder_id'].tolist()
    train_dataset = GestureDataset(CFG.TRAIN_DIR, CFG.MASK_TRAIN_DIR, train_folder_ids, CFG.FRAMES, train_transform)
    valid_dataset = GestureDataset(CFG.TRAIN_DIR, CFG.MASK_TRAIN_DIR, valid_folder_ids, CFG.FRAMES, valid_transform)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=CFG.BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=CFG.BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False)
    device = CFG.DEVICE
    logger.info(f'training with {device}')
    model = GestureSeqModel(CFG.BACKBONE, backbone_pretrained=CFG.PRETRAINED, lstm_dim=CFG.LSTM_HIDDEN_SIZE, lstm_layers=CFG.LSTM_LAYERS, n_classes=CFG.NUM_CLASSES)
    model.to(device)
    logger.info(f'backbone: {CFG.BACKBONE}')
    logger.info(f'drop rate: {CFG.DROP_RATE} | drop path rate: {CFG.DROP_PATH_RATE}')
    optimizer = Adam(model.parameters(), lr=CFG.BASE_LR, weight_decay=CFG.WEIGHT_DECAY, amsgrad=False)
    scheduler = get_scheduler(optimizer)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=CFG.WARM_UP, after_scheduler=scheduler)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    best_valid_loss = 9999
    best_valid_ep = 0
    patient = CFG.PATIENCE
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, CFG.NUM_EPOCHS+1):
        scheduler_warmup.step(epoch)

        # =============== Training ==============
        train_loss, train_acc = train_valid_fn(train_loader,model,criterion, scaler, optimizer=optimizer,device=device,
                                    scheduler=scheduler,epoch=epoch,mode='train', metric='acc')
        valid_loss, valid_acc = train_valid_fn(valid_loader,model,criterion, scaler, device=device,epoch=epoch,mode='valid', metric='acc')
        valid_labels = valid_df[CFG.TARGET_COL].values
        
        logger.info('Train loss: %.4f | Train accuracy: %.3f | Valid loss: %.4f | Valid accuracy %.3f'%(train_loss, train_acc, valid_loss, valid_acc))
        
        if(valid_loss < best_valid_loss):
            best_valid_loss = valid_loss
            best_valid_ep = epoch
            patience = CFG.PATIENCE # reset patient

            # save model
            name = os.path.join(CFG.OUTPUT_DIR, f'%s_fold%d_best.pth'%(CFG.BACKBONE, fold))
            logger.info('Saving model to: ' + name)
            torch.save({'model': model.state_dict(), 'best': valid_acc}, name)
        else:
            patience -= 1
            logger.info('Decrease early-stopping patient by 1 due valid loss not decreasing. Patience='+ str(patience))

        if(patience == 0):
            logger.info('Early stopping patient = 0. Early stop')
            break
            
    checkpoint = torch.load(os.path.join(CFG.OUTPUT_DIR, f'%s_fold%d_best.pth'%(CFG.BACKBONE, fold)))
    best_acc = checkpoint['best']
    return best_acc

def infer_fn(dataloader, model, checkpoint_dir=None, device='cuda:0'):
    logger.info('Starting infer on testset ...')
    model.to(device)
    model.eval()
    
    if checkpoint_dir is None or checkpoint_dir == '':
        logger.warning('Checkpoint not found!')
        return None
    else:
        model.load_state_dict(torch.load(checkpoint_dir)['model'])
        tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
        all_predictions = []
        for i, batch in tk0:
            # input, gt
            voxels = batch
            voxels = voxels.to(device)
            # prediction
            with torch.no_grad():
                logits = model(voxels)
                probs = logits.softmax(1).detach().cpu().numpy()
#                 preds = probs.argmax(1)
            all_predictions.append(probs)
        all_predictions = np.concatenate(all_predictions)
        return all_predictions 

def test_fn(dataloader, model, checkpoint_dir=None, device='cuda:0'):
    logger.info('Starting testing on testset ...')
    model.to(device)
    model.eval()
    
    if checkpoint_dir is None or checkpoint_dir == '':
        logger.warning('Checkpoint not found!')
        return None
    else:
        model.load_state_dict(torch.load(checkpoint_dir)['model'])
        tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
        all_predictions = []
        for i, batch in tk0:
            # input, gt
            voxels = batch
            voxels = voxels.to(device)
            # prediction
            with torch.no_grad():
                logits = model(voxels)
                probs = logits.softmax(1).detach().cpu().numpy()
                preds = probs.argmax(1)
            all_predictions.append(preds)
        all_predictions = np.concatenate(all_predictions)
        return all_predictions 
		



