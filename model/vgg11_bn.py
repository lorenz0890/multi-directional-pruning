import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn

def VGG11BN(num_classes=10):
    model = nn.Sequential(
            torchvision.models.vgg11_bn(num_classes=10),
            nn.LogSoftmax(1)
        )
    return model