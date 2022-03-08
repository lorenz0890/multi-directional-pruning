from torchvision import datasets, transforms
import torch

from experiments import ADMMRetrain, ADMMIntra, GDTopK, MCGDTopK, MCGDTopKACDK, MCGDTopKAC, Baseline, \
    MCGDTopKACDKADMMIntra, REPruningMCGDTopKACDKADMMIntra, REPruningAC, MCGDTopKACDKADMMRetrain, REPruningADMMIntra, \
    REPruningACADMMIntra, REPruningACADMMRetrain
from experiments import REPruning
from experiments.re_pruning_admm_retrain import REPruningADMMRetrain
from utils import DataFactory, ModelFactory
from utils import Logger
from utils.visualization import Visualization


class ExperimentFactory:
    def __init__(self):
        self.data_factory = DataFactory()
        self.model_factory = ModelFactory()

    def get_experiment(self, config):
        experiment = None
        train_loader, test_loader = self.data_factory.get_dataset(config)
        model = self.model_factory.get_model(config)
        logger, visualization = Logger(config.get_raw()), Visualization(config.get_raw())
        if config.get('EXPERIMENT', 'name', str) == 'admm_retrain':
            experiment = ADMMRetrain(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 'admm_intra':
            experiment = ADMMIntra(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k':
            experiment = GDTopK(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k_mc':
            experiment = MCGDTopK(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k_mc_ac':
            experiment = MCGDTopKAC(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k_mc_ac_dk':
            experiment = MCGDTopKACDK(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 're_pruning':
            experiment = REPruning(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 're_pruning_ac':
            experiment = REPruningAC(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 're_pruning_admm_intra':
            experiment = REPruningADMMIntra(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 're_pruning_admm_retrain':
            experiment = REPruningADMMRetrain(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 're_pruning_ac_admm_intra':
            experiment = REPruningACADMMIntra(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 're_pruning_ac_admm_retrain':
            experiment = REPruningACADMMRetrain(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 'baseline':
            experiment = Baseline(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k_mc_ac_dk_admm_intra':
            experiment = MCGDTopKACDKADMMIntra(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 'gd_top_k_mc_ac_dk_admm_retrain':
            experiment = MCGDTopKACDKADMMRetrain(model, train_loader, test_loader, config, logger, visualization)
        if config.get('EXPERIMENT', 'name', str) == 're_pruning_gd_top_k_mc_ac_dk_admm_intra':
            experiment = REPruningMCGDTopKACDKADMMIntra(model, train_loader, test_loader, config, logger, visualization)
        return experiment