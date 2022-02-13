import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn

def MobileNetV2(num_classes=10):
    model = nn.Sequential(
            torchvision.models.mobilenet_v2(num_classes=num_classes),
            nn.LogSoftmax(1)
        )
    return model

