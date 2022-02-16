import torch.nn as nn
import torch.nn.functional as F
import torchvision


def AlexNet(num_classes=10):
    model = nn.Sequential(
            torchvision.models.alexnet(num_classes=num_classes),
            nn.LogSoftmax(1)
        )
    return model
