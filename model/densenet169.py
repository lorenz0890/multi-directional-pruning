import torch
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn

def DenseNet169(num_classes=10):
    model = nn.Sequential(
            torchvision.models.DenseNet(num_classes=num_classes,
                                        growth_rate=32,
                                        block_config=(6, 12, 32, 32),
                                        num_init_features=64
                                        ),
            nn.LogSoftmax(1)
    )
    return model
