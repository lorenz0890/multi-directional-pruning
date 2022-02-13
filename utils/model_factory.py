import math

import torchvision.models
from torch import nn
from torchvision import datasets, transforms
import torch

from model import LeNet, AlexNet, VGG16, ResNet18, AlexNet_S, VGG16BN, VGG13, VGG13BN, VGG11, VGG11BN, MobileNetV2, \
    MobileNetV3_S, MobileNetV3_L
from .custom_init import truncated_normal_variance_scaling

class ModelFactory:
    def __init__(self):
        pass

    def get_model(self, config):
        use_cuda = not config.get('OTHER', 'no_cuda', bool) and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = None
        if config.get('EXPERIMENT', 'model', str) == 'lenet':
            model = LeNet().to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'alexnet':
            model = AlexNet(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'alexnet_s':
            model = AlexNet_S(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'vgg16':
            model = VGG16(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'vgg16_bn':
            model = VGG16BN(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'vgg13':
            model = VGG13(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'vgg13_bn':
            model = VGG13BN(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'vgg11':
            model = VGG11(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'vgg11_bn':
            model = VGG11BN(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'resnet18':
            model = ResNet18(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'mobilenet_v2':
            model = MobileNetV2(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'mobilenet_v3_s':
            model = MobileNetV3_S(config.get('SPECIFICATION', 'num_classes', int)).to(device)
        elif config.get('EXPERIMENT', 'model', str) == 'mobilenet_v3_l':
            model = MobileNetV3_L(config.get('SPECIFICATION', 'num_classes', int)).to(device)

        '''
        def init_weights(m):
            if hasattr(m, 'weight') and len(m.weight.shape) > 1:
                truncated_normal_variance_scaling(m.weight, 0.5)
            if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.bias.data.uniform_(-stdv, stdv)

        model.apply(init_weights)
        '''
        return model