import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn
from .wrn import WideResNet

def WRN16_10(num_classes=10):
    model = nn.Sequential(
            WideResNet(depth=16, num_classes=num_classes, widen_factor=10),
            nn.LogSoftmax(1)
        )
    return model