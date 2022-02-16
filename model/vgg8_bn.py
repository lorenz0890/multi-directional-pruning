import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn
from .vgg import vgg8_bn

def VGG8BN(num_classes=10):
    model = nn.Sequential(
            vgg8_bn(num_classes=num_classes),
            nn.LogSoftmax(1)
        )
    return model