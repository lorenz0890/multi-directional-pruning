import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn

def ResNet18():
    model = nn.Sequential(
            torchvision.models.resnet18(num_classes=10),
            nn.LogSoftmax(1)
    )
    return model

