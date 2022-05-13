import torch.nn as nn

from .resnet import resnet32


def ResNet32(num_classes=10):
    model = nn.Sequential(
        resnet32(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
