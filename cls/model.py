import torch
import torch.nn as nn
import os
import timm
from config import CFG 

class GestureSeqModel(nn.Module):
    def __init__(self, backbone_name, backbone_pretrained,
                 lstm_dim=64, lstm_layers=1, lstm_dropout=CFG.LSTM_DROP, 
                 n_classes=1):
        super(GestureSeqModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=backbone_pretrained, drop_rate=CFG.DROP_RATE, drop_path_rate=CFG.DROP_PATH_RATE)
#         self.backbone.load_state_dict(torch.load(backbone_pretrained))
        if "resnext50_32x4d" in backbone_name:
            lstm_inp_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name.startswith("res"):
            lstm_inp_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone_name.startswith("tf_efficientnet"):
            lstm_inp_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif backbone_name.startswith("vit_"):
            lstm_inp_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif backbone_name.startswith("eca_"):
            lstm_inp_dim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        elif backbone_name.startswith("dm_"):
            lstm_inp_dim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        elif backbone_name.startswith("nfnet_"):
            lstm_inp_dim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        elif backbone_name.startswith("swin_"):
            lstm_inp_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif backbone_name.startswith("cait_"):
            lstm_inp_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif backbone_name.startswith("densenet"):
            lstm_inp_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        
        self.lstm = nn.LSTM(lstm_inp_dim, lstm_dim, num_layers=lstm_layers, 
                            batch_first=True, bidirectional=True,
                            dropout=lstm_dropout)
        
        self.clf_head = nn.Linear(lstm_dim*2*CFG.SEQ_LEN, n_classes)
        
    def forward(self, x):
        n = x.shape[0]
        seq_length = x.shape[1]
        concat_x = torch.cat([x[i] for i in range(n)], axis=0)
        concat_x = self.backbone(concat_x)
        
        
        stacked_x = torch.stack([concat_x[i*seq_length:i*seq_length+seq_length] for i in range(n)], axis=0)
        
        seq_features, _ = self.lstm(stacked_x)
        seq_features = seq_features.reshape(n,-1)
        
        logits = self.clf_head(seq_features)
        
        return logits