import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn
from .resnet import resnet20

def ResNet20(num_classes=10):
    model = nn.Sequential(
            resnet20(num_classes=num_classes),
            nn.LogSoftmax(1)
        )
    return model