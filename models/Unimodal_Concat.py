import torch
import torch.nn as nn


def Unimodal_Concat(nn.Module):
    def __init__(self,
                vision_module,
                lang_module,
                vis_dim,
                lang_dim,
                cat_dim,
                num_classes=2,
                dropout=0.0        
        )
        super(Unimodal_Concat, self)__init__()
        self.vision_module = vision_module
        self.lang_module = lang_module
        self.fc1 = nn.Linear(vis_dim+lang_dim, cat_dim)
        self.fc2 = nn.Linear(cat_dim, num_classes)
        self.dropout(dropout)
    
    def forward(self, img, txt, label=None):
        image_features = nn.ReLU(vision_module(img))
        txt_features = nn.ReLU(lang_module(txt))
        cat = torch.cat([image_features,txt_features], dim=1)
        fc1 = self.dropout(nn.ReLU(self.fc1(cat)))  
        pred = self.fc2(fc1)
        return pred