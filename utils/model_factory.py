from torchvision import datasets, transforms
import torch

from model import LeNet, AlexNet


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
            model = AlexNet().to(device)
        return model