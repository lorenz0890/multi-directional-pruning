import torch.nn as nn
import torchvision.models


def ResNet18(num_classes=10):
    model = nn.Sequential(
        torchvision.models.resnet18(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
