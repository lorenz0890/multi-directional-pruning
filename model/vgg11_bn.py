import torch.nn as nn
import torchvision.models



def VGG11BN(num_classes=10):
    model = nn.Sequential(
        torchvision.models.vgg11_bn(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
