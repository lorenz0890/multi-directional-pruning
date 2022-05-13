import torch.nn as nn
import torchvision.models


def VGG13BN(num_classes=10):
    model = nn.Sequential(
        torchvision.models.vgg13_bn(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
