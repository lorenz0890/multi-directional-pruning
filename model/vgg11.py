import torch.nn as nn
import torchvision.models



def VGG11(num_classes=10):
    model = nn.Sequential(
        torchvision.models.vgg11(num_classes=num_classes),
        nn.LogSoftmax(1)
    )
    return model
