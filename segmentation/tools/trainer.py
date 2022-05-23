import torch
import multiprocessing as mp
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP 
from utilizes.utils import setup_ddp
from utilizes.utils import clip_gradient, AvgMeter
import timeit
import os
import numpy as np
from auxiliary.metrics.metrics import *
import datetime

class Trainer:
    def __init__(self, model, model_name, optimizer, loss, scheduler, save_dir, save_from, logger, device, use_amp=False, use_ddp=False, multi_loss=False, name_writer=None):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler= scheduler
        self.save_from = save_from
        self.save_dir = save_dir
        self.logger = logger
        self.device = device
        self.use_amp = use_amp
        self.use_ddp = use_ddp
        self.name_writer = name_writer
        if self.name_writer == None:
            self.writer = SummaryWriter()
        else:
            save_wr = f'./runs/{self.name_writer}'
            self.writer = SummaryWriter(save_wr)
        
        self.scaler = GradScaler(enabled=self.use_amp)
        self.multi_loss = multi_loss
    
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def train_loop(self, train_loader, val_loader, num_epochs, img_size=352, size_rates=[1], clip_grad=0.5, is_val=False):
        start = timeit.default_timer()
        best_score = 0
        for epoch in range(num_epochs):
            self.model.train()
            if self.use_ddp:
                self.model = DDP(self.model, device_ids=[setup_ddp()])
            train_loss = 0.0
            total_iters = len(train_loader)
            pbar = tqdm(enumerate(train_loader), total=total_iters, 
                        desc=f"Epoch: [{epoch + 1}/{num_epochs}] | Iter: [{0}/{total_iters}] | LR: {self.get_lr() :.8f}")
            for iter, (images, gts) in pbar:
                for rate in size_rates:
                    self.optimizer.zero_grad()
                    images = images.to(self.device)
                    gts = gts.to(self.device)
                    
                    trainsize = int(round(img_size * rate / 32) * 32)
                    if rate != 1:
                        images = torch.nn.functional.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        gts = torch.nn.functional.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        
                    if self.use_amp: 
                        
                        with autocast(enabled=self.use_amp):
                            logits = self.model(images)
                            loss = self.loss(logits, gts)
                        self.scaler.scale(loss).backward()
                        clip_gradient(self.optimizer, clip_grad)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        torch.cuda.synchronize()
                        
                    else:
                        if self.multi_loss == False:
                            logits = self.model(images)
                            loss = self.loss(logits, gts)
                            
                            loss.backward()
                            clip_gradient(self.optimizer, clip_grad)
                            self.optimizer.step()  
                        elif self.multi_loss == True:
                            logits1, logits2, logits3, logits4 = self.model(images)
                            loss = self.loss(logits1, gts) + self.loss(logits2, gts) + self.loss(logits3, gts) + self.loss(logits4, gts)
                            
                            loss.backward()
                            clip_gradient(self.optimizer, clip_grad)
                            self.optimizer.step()
                        
                    if rate == 1:
                        train_loss += loss.item()
                        self.writer.add_scalar('train/loss', train_loss, epoch)
                    
                    pbar.set_description(f'Epoch: [{epoch + 1}/ {num_epochs}] | Iter: [{0}/{total_iters}] | LR: {self.get_lr():.6f}')
            train_loss /= iter + 1
            os.makedirs(self.save_dir, exist_ok=True)
            if is_val == True:
                iou_score = self.val_loop(val_loader, epoch)
                if iou_score > best_score:
                    self.logger.info(
                    "[Saving Snapshot:]"
                    + os.path.join(
                        self.save_dir, self.model_name + "_best.pth"
                    ))
                    torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    os.path.join(
                        self.save_dir, self.model_name + "_best.pth"
                    ),
                    
                    )
                    best_score = iou_score
                
            
            if is_val == False:
                self.logger.info(f'Epoch: [{epoch + 1}/ {num_epochs}] | Train loss: [{train_loss}]')
                
            if epoch >= self.save_from and (epoch + 1) % 1 == 0 or epoch == 2:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    os.path.join(
                        self.save_dir, self.model_name + "_%d.pth" % (epoch + 1)
                    ),
                )
                self.logger.info(
                    "[Saving Snapshot:]"
                    + os.path.join(
                        self.save_dir, self.model_name + "_%d.pth" % (epoch + 1)
                    )
                )    
            self.scheduler.step(epoch)
        end = timeit.default_timer()

        self.logger.info("Training cost: " + str(end - start) + "seconds")
            
            
    def val_loop(self, val_loader, epoch):
        len_val = len(val_loader)
        
        tp_all = 0
        fp_all = 0
        fn_all = 0
        tn_all = 0

        tpr = 0
        fpr = 0
        fnr = 0
        tnr = 0


        mean_precision = 0
        mean_recall = 0
        mean_iou = 0
        mean_dice = 0

        mean_F2 = 0
        mean_acc = 0

        val_loss = AvgMeter()
        images = []
        for i, pack in enumerate(val_loader, start=1):
            image, gt, gt_resize = pack
            self.model.eval()

            gt_ = gt.cuda()
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)

            res2 = 0
            image_ = image
            image = image.cuda()
            gt_resize = gt_resize.cuda()

            res = self.model(image)
            res = torch.nn.functional.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
            loss2 = self.loss(res, gt_)
            val_loss.update(loss2.data, 1)
            self.writer.add_scalar(
                "Val_loss", val_loss.show(), epoch * len(val_loader) + i
            )
            if i == len_val - 1:
                
                self.logger.info(
                    "Valid | Epoch [{:03d}/{:03d}], with lr = {}, Step [{:04d}],\
                    [val_loss: {:.4f}]".format(
                        epoch,
                        epoch,
                        self.optimizer.param_groups[0]["lr"],
                        i,
                        val_loss.show(),
                    )
                )
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            if i <= 0:
                res2 = res2.data.cpu().numpy().round()[0][0]
                gt_resize = gt_resize.data.cpu().numpy()[0][0]
                mask_img = (
                    np.asarray(image_.data.cpu().numpy()[0])
                    + 180
                    * np.array(
                        (
                            np.zeros_like(res2),
                            res2,
                            np.zeros_like(res2),
                        )
                    ).transpose((0, 1, 2))
                    + 180
                    * np.array(
                        (gt_resize, np.zeros_like(gt_resize), np.zeros_like(gt_resize))
                    ).transpose((0, 1, 2))
                )
                images.append(mask_img)

            pr = res.round()
            tp = np.sum(gt * pr)
            fp = np.sum(pr) - tp
            fn = np.sum(gt) - tp
            tn = np.sum((1 - pr) * (1 - gt))

            tpr += tp/(tp+fn)
            fpr += fp/(fp+tn)
            fnr += fn/(fn+tp)
            tnr += tn/(tn+fp)

            tp_all += tp
            fp_all += fp
            fn_all += fn

            mean_precision += precision_m(gt, pr)
            mean_recall += recall_m(gt, pr)
            mean_iou += jaccard_m(gt, pr)
            mean_dice += dice_m(gt, pr)
            mean_F2 += (5 * precision_m(gt, pr) * recall_m(gt, pr)) / (
                4 * precision_m(gt, pr) + recall_m(gt, pr)
            )

        mean_precision /= len_val
        mean_recall /= len_val
        mean_iou /= len_val
        mean_dice /= len_val
        mean_F2 /= len_val
        tpr /= len_val
        fpr /= len_val
        fnr /= len_val
        tnr /= len_val

        self.logger.info(
            "Macro scores: Dice: {:.3f} | IOU: {:.3f} | Precision: {:.3f} | Recall: {:.3f} | F2: {:.3f}".format(
                mean_dice, mean_iou, mean_precision, mean_recall, mean_F2
            )
        )

        self.writer.add_scalar("mean_dice", mean_dice, epoch)
        self.writer.add_scalar("mean_iou", mean_iou, epoch)
        self.writer.add_scalar("tpr", tpr, epoch)
        self.writer.add_scalar("fpr", fpr, epoch)
        self.writer.add_scalar("fnr", fnr, epoch)
        precision_all = tp_all / (tp_all + fp_all + 1e-07)
        recall_all = tp_all / (tp_all + fn_all + 1e-07)
        dice_all = 2 * precision_all * recall_all / (precision_all + recall_all)
        iou_all = (
            recall_all
            * precision_all
            / (recall_all + precision_all - recall_all * precision_all)
        )
        self.logger.info(
            "Micro scores: Dice: {:.3f} | IOU: {:.3f} | Precision: {:.3f} | Recall: {:.3f}".format(
                dice_all, iou_all, precision_all, recall_all
            )
        )
        self.writer.add_scalar("dice_all", dice_all, epoch)
        self.writer.add_scalar("iou_all", iou_all, epoch)
        
        return mean_iou
        
