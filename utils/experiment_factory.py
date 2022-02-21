from torchvision import datasets, transforms
import torch

from experiments import ADMMRetrain, ADMMIntra, GDTopK, MCGDTopK, MCGDTopKACDK, MCGDTopKAC, Baseline, \
    MCGDTopKACDKADMMIntra, REPruningMCGDTopKACDKADMMIntra
from experiments import REPruning
from utils import DataFactory, ModelFactory
from utils import Logger
from utils.vis import Visualization


class ExperimentFactory:
    def __init__(self):
        self.data_factory = DataFactory()
        self.model_factory = ModelFactory()

    def get_experiment(self, config):
        experiment = None
        train_loader, test_loader = self.data_factory.get_dataset(config)
        model = self.model_factory.get_model(config)
        if config.get('EXPERIMENT', 'name', str) == 'admm_retrain':
            experiment = ADMMRetrain(model, train_loader, test_loader, config, Logger(config.get_raw()))
        if config.get('EXPERIMENT', 'name', str) == 'admm_intra':
            experiment = ADMMIntra(model, train_loader, test_loader, config, Logger(config.get_raw()))
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k':
            experiment = GDTopK(model, train_loader, test_loader, config, Logger(config.get_raw()))
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k_mc':
            experiment = MCGDTopK(model, train_loader, test_loader, config, Logger(config.get_raw()))
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k_mc_ac':
            experiment = MCGDTopKAC(model, train_loader, test_loader, config, Logger(config.get_raw()))
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k_mc_ac_dk':
            experiment = MCGDTopKACDK(model, train_loader, test_loader, config, Logger(config.get_raw()))
        if config.get('EXPERIMENT', 'name', str) == 're_pruning':
            #experiment = REPruning(model, train_loader, test_loader, config, Logger(config.get_raw()))
            experiment = REPruning(model, train_loader, test_loader, config,
                                   Logger(config.get_raw()), Visualization(config.get_raw()))
        if config.get('EXPERIMENT', 'name', str) == 'baseline':
            experiment = Baseline(model, train_loader, test_loader, config, Logger(config.get_raw()))
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k_mc_ac_dk_admm_intra':
            experiment = MCGDTopKACDKADMMIntra(model, train_loader, test_loader, config, Logger(config.get_raw()))
        if config.get('EXPERIMENT', 'name', str) == 're_pruning_gd_top_k_mc_ac_dk_admm_intra':
            experiment = REPruningMCGDTopKACDKADMMIntra(model, train_loader, test_loader, config, Logger(config.get_raw()))

        return experiment