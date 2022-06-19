import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn


def MobileNetV3_S(num_classes=10):
    model = nn.Sequential(
        torchvision.models.mobilenet_v3_small(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
