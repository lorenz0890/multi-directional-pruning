import torch.nn as nn

from .resnet import resnet56


def ResNet56(num_classes=10):
    model = nn.Sequential(
        resnet56(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
