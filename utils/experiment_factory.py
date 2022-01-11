from torchvision import datasets, transforms
import torch

from experiments import ADMMRetrain
from model import LeNet, AlexNet
from utils import DataFactory, ModelFactory


class ExperimentFactory:
    def __init__(self):
        self.data_factory = DataFactory()
        self.model_factory = ModelFactory()

    def get_experiment(self, config):
        experiment = None
        train_loader, test_loader = self.data_factory.get_dataset(config)
        model = self.model_factory.get_model(config)
        if config.get('EXPERIMENT', 'name', str) == 'admm_retrain':
            experiment = ADMMRetrain(model, train_loader, test_loader, config)

        return experiment