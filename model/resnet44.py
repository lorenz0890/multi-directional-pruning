import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn
from .resnet import resnet44

def ResNet44(num_classes=10):
    model = nn.Sequential(
            resnet44(num_classes=num_classes),
            nn.LogSoftmax(1)
        )
    return model