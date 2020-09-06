import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms

class ResNet_module(nn.Module):
    def __init__(self, output_dim, device='cpu', learn=False):
        super(ResNet_module, self).__init__()
        self.model = models.resnet18(pretrained=True)
        if not learn:
            for param in self.model.parameters():
                param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, output_dim)
        self.model.to(device)

    def forward(self, img):
        emb = self.model(img)
        return emb.squeeze()
    
    @staticmethod
    def transform(img_dim):
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    size=(img_dim, img_dim)
                ),        
                torchvision.transforms.ToTensor(),
                # all torchvision models expect the same
                # normalization mean and std
                # https://pytorch.org/docs/stable/torchvision/models.html
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return image_transform