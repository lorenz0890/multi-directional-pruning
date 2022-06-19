import torch.nn as nn

from .vgg import vgg8


def VGG8(num_classes=10):
    model = nn.Sequential(
        vgg8(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
