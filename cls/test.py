from config import CFG 
import torch
import torch.nn as nn
from utils import *
from trainer import train_valid_fn, test_fn
from dataset import GestureTestDataset, DataLoader, Dataset
from model import *


train_transform = get_train_transforms()
valid_transform = get_valid_transforms()
test_folder_list = os.listdir(CFG.TEST_DIR)
test_folder_list = [int(i) for i in test_folder_list]
test_folder_list.sort()
test_dataset = GestureTestDataset(CFG.TEST_DIR, CFG.MASK_TEST_DIR, test_folder_list, CFG.FRAMES, transform=valid_transform)
test_loader = DataLoader(test_dataset, 
                              batch_size=CFG.BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False)
device = CFG.DEVICE
ckpt = '/home/s/hungpv/polyps/code/bn_code/cls/checkpoint/dm_nfnet_f0_v1/dm_nfnet_f0_fold3_best.pth'
model = GestureSeqModel(CFG.BACKBONE, backbone_pretrained=False, lstm_dim=CFG.LSTM_HIDDEN_SIZE, lstm_layers=CFG.LSTM_LAYERS, n_classes=CFG.NUM_CLASSES)
preds = test_fn(test_loader, model, checkpoint_dir=ckpt, device='cuda:0')

if __name__ == '__main__':
    with open('/home/s/hungpv/polyps/code/bn_code/dataset/private.txt', 'r') as f:
        labels = f.read().splitlines()
    f.close()
    labels = [CFG.LABELS.index(i) for i in labels]
    test_score = get_score(labels, preds)
    print('Test score: ',test_score)

