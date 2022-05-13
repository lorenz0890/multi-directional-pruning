import torch.nn as nn
import torchvision.models


def ResNet50(num_classes=10):
    model = nn.Sequential(
        torchvision.models.resnet50(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
