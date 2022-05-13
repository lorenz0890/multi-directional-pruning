import torch.nn as nn
import torchvision.models


def DenseNet161(num_classes=10):
    model = nn.Sequential(
        torchvision.models.DenseNet(num_classes=num_classes,
                                    growth_rate=48,
                                    block_config=(6, 12, 36, 24),
                                    num_init_features=96
                                    ),
        nn.LogSoftmax(1)
    )
    return model
