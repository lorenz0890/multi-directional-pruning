import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn

def VGG16():
    model = nn.Sequential(
            torchvision.models.vgg16(num_class=10),
            nn.LogSoftmax(1)
        )
    return model

