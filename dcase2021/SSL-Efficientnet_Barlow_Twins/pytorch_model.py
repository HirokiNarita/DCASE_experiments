import torch
from torch import nn
import timm

class EfficientNet_b1(nn.Module):
    def __init__(self, n_out):
        super(EfficientNet_b1, self).__init__()
        #モデルの定義
        self.effnet = timm.create_model('efficientnet_b1', pretrained=True)
        #最終層の再定義
        self.effnet.classifier = nn.Linear(1280, 1280)

    def forward(self, x):
        return self.effnet(x)
