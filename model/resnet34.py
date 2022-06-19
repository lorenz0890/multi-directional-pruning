import torch.nn as nn
import torchvision.models



def ResNet34(num_classes=10):
    model = nn.Sequential(
        torchvision.models.resnet34(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
