import torch
import torch.nn as nn


def Unimodal_Concat(nn.Module):
    def __init__(self,
                vision_module,
                lang_module,
                vis_dim,
                lang_dim,
                cat_dim,
                loss_func,
                num_classes=2,
                dropout=0.0        
        )
        super(Unimodal_Concat, self)__init__()
        self.vision_module = vision_module
        self.lang_module = lang_module
        self.fc1 = nn.Linear(vis_dim+lang_dim, cat_dim)
        self.fc2 = nn.Linear(cat_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.loss_func = loss_func
    
    def forward(self, img, txt, label=None):
        image_features = nn.ReLU(vision_module(img))
        txt_features = nn.ReLU(lang_module(txt))
        cat = torch.cat([image_features,txt_features], dim=1)
        fc1 = self.dropout(nn.ReLU(self.fc1(cat)))  
        pred = self.fc2(fc1)
        loss = loss_func(pred, label) if label is not None else None
        return (pred, loss)