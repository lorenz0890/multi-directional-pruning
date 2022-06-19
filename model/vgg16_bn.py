import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn


def VGG16BN(num_classes=10):
    model = nn.Sequential(
        torchvision.models.vgg16_bn(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
