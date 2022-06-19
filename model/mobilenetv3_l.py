import torch.nn as nn
import torchvision.models



def MobileNetV3_L(num_classes=10):
    model = nn.Sequential(
        torchvision.models.mobilenet_v3_large(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
